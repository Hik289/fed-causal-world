"""
exp1_alfworld.py — Exp1 on ALFWorld valid_unseen, 9 baselines × N tasks × 1 seed.

specification: PILOT FIRST (5 tasks × B2 only), measure cost/task, abort if
estimate $0.025 ±20% violated. Then full 268 tasks × 9 baselines on PASS.

ReAct-style agent (no tool-calling, plain text actions):
  System: baseline-specific WM header + ALFWorld instructions
  User:   observation + admissible commands (truncated to 30)
  Model:  reasoning + "ACTION: <command>"
  Loop until won/lost or max_steps=30.

Differences from τ-bench harness:
  - No user simulator (env is deterministic)
  - No tool-calling (plain text command)
  - Admissible-command grounding (model picks from finite list)
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
import traceback
import yaml
from typing import Any, Dict, List, Optional

# API configuration is read from the environment.
AZURE_API_KEY = os.environ.get("FED_CAUSAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
AZURE_API_BASE = os.environ.get("FED_CAUSAL_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
MODEL_NAME = os.environ.get("FED_CAUSAL_MODEL", "openai/gpt-5.4-mini")
DEPLOYMENT = MODEL_NAME.rsplit("/", 1)[-1]
PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00

import litellm
_original_completion = litellm.completion
_usage_log: List[Dict[str, Any]] = []


def _patched_completion(*args, **kwargs):
    if AZURE_API_KEY:
        kwargs.setdefault("api_key", AZURE_API_KEY)
    if AZURE_API_BASE:
        kwargs.setdefault("api_base", AZURE_API_BASE)
    if not kwargs.get("model", "").endswith(DEPLOYMENT):
        kwargs["model"] = MODEL_NAME
    kwargs.setdefault("custom_llm_provider", "openai")
    t0 = time.time()
    res = _original_completion(*args, **kwargs)
    elapsed = time.time() - t0
    try:
        u = res.usage
        pt = getattr(u, "prompt_tokens", 0) or 0
        ct = getattr(u, "completion_tokens", 0) or 0
        usd = (pt * PRICE_INPUT_PER_1M + ct * PRICE_OUTPUT_PER_1M) / 1_000_000.0
        if hasattr(res, "_hidden_params"):
            res._hidden_params["response_cost"] = usd
        else:
            res._hidden_params = {"response_cost": usd}
        _usage_log.append({"prompt_tokens": pt, "completion_tokens": ct,
                           "usd": usd, "elapsed_s": elapsed})
    except Exception:
        pass
    return res


litellm.completion = _patched_completion


# --------------------------------------------------------------------------
# ALFWorld modularization summary (from modularization_spec.md §2)
# --------------------------------------------------------------------------

MODULARIZATION_SUMMARY = (
    "ALFWorld has 6 functional modules: navigation, container_access, "
    "object_manipulation, appliance, object_property, task_monitor. "
    "Cross-module dependencies (validated edges): navigation→container_access, "
    "container_access→object_manipulation, navigation→object_manipulation, "
    "object_manipulation→appliance, appliance→object_property, "
    "object_manipulation→object_property, object_property→task_monitor, "
    "object_manipulation→task_monitor."
)

ORACLE_EDGES = (
    "GROUND-TRUTH causal interface edges (Oracle): "
    "navigation→container_access (agent_at unlocks visible_inside), "
    "container_access→object_manipulation (container_opened enables pick), "
    "navigation→object_manipulation (agent_at enables pick at receptacle), "
    "object_manipulation→appliance (insert object enables effect), "
    "appliance→object_property (effect_applied flips property after timer), "
    "object_manipulation→object_property (place in sink → is_clean=1), "
    "object_property→task_monitor (property_changed satisfies predicate), "
    "object_manipulation→task_monitor (place satisfies in() predicate)."
)

BASELINE_HEADERS = {
    "B0_NoWM": (
        "WORLD MODEL: None. Act greedily. Do not predict future states."
    ),
    "B1_LocalWM": (
        "WORLD MODEL: Per-module LOCAL only. Track navigation, container "
        "state, holding state separately. Do not model cross-module effects."
    ),
    "B2_GlobalSeqWM": (
        "WORLD MODEL: GLOBAL SEQUENCE. Track the global event sequence "
        "(navigation, container open/close, pick/place, appliance toggle, "
        "property changes) and predict next states from this sequence. No "
        "explicit causal structure."
    ),
    "B3_TransitionGraph": (
        "WORLD MODEL: GLOBAL TRANSITION GRAPH. Maintain (state, action) → "
        "next_state transitions for the world state. No causal direction."
    ),
    "B4_CorrelationWM": (
        "WORLD MODEL: CORRELATION-BASED. Track which events co-occur and "
        "use co-occurrence to predict next steps."
    ),
    "B5_TemporalWM": (
        "WORLD MODEL: TEMPORAL PRECEDENCE. Track event order and lag. "
        "For example, toggling an appliance takes 1-3 steps before properties "
        "change. Use temporal patterns to plan."
    ),
    "B7_CausalWMNoInt": (
        "WORLD MODEL: CAUSAL graph from OBSERVATIONAL data only. "
        + MODULARIZATION_SUMMARY +
        " The graph was learned from observation only; some edges may be "
        "directionally ambiguous."
    ),
    "B8_FedCausalCompose": (
        "WORLD MODEL: FedCausalCompose — Full causal graph + decentralized "
        "causal control. " + MODULARIZATION_SUMMARY +
        " For each action: INSPECT required upstream states (e.g. agent_at "
        "before pick), PREDICT downstream effects (e.g. appliance timer → "
        "property change), BLOCK action if upstream not satisfied, VERIFY "
        "outcomes. The graph is validated by intervention-response matching."
    ),
    "B10_OracleCausalWM": (
        "WORLD MODEL: ORACLE causal graph. " + ORACLE_EDGES +
        " Use this oracle graph for causal rollout."
    ),
}


SYSTEM_INSTRUCTIONS = """You are an agent in an ALFWorld text environment.
Your task is described in the first observation. To act, output your next
command. Be CONCISE.

OUTPUT FORMAT:
Reasoning: <one short sentence>
ACTION: <exact command from the admissible commands list>

Rules:
- ACTION must match an item from the admissible commands list exactly.
- Do NOT invent commands.
- If the task is done, output 'ACTION: look' (no-op)."""


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

ACTION_RE = re.compile(r"ACTION:\s*(.+?)$", re.IGNORECASE | re.MULTILINE)


def extract_action(text: str, admissible: List[str]) -> str:
    """Parse ACTION: <cmd>; fallback to first admissible if unparseable."""
    m = ACTION_RE.search(text)
    if m:
        cmd = m.group(1).strip().rstrip(".")
        # Match exactly first
        if cmd in admissible:
            return cmd
        # Loose match: substring
        for adm in admissible:
            if cmd.lower() == adm.lower():
                return adm
        for adm in admissible:
            if cmd.lower() in adm.lower() or adm.lower() in cmd.lower():
                return adm
    # Fallback: pick first non-look admissible
    for adm in admissible:
        if not adm.startswith("look"):
            return adm
    return admissible[0] if admissible else "look"


def run_one_task(baseline_id: str, env, max_steps: int = 30) -> Dict[str, Any]:
    """env is an already-initialized AlfredTWEnv batch_size=1."""
    header = BASELINE_HEADERS[baseline_id]
    obs, info = env.reset()
    obs_text = obs[0]
    admissible = info.get("admissible_commands", [[]])[0]
    won = False
    n_calls = 0
    history_actions = []

    messages = [
        {"role": "system", "content": f"{header}\n\n{SYSTEM_INSTRUCTIONS}"},
        {"role": "user", "content": f"Observation:\n{obs_text}\n\nAdmissible commands (pick one):\n"
                                    + "\n".join(f"- {c}" for c in admissible[:40])
                                    + "\n\nWhat's your next action?"},
    ]

    for step in range(max_steps):
        try:
            res = litellm.completion(
                model=MODEL_NAME, messages=messages,
                max_tokens=128, temperature=0.0,
            )
            n_calls += 1
            text = res.choices[0].message.content or ""
            action = extract_action(text, admissible)
        except Exception as exc:
            return {"error": f"LLM error step {step}: {exc}", "won": False,
                    "n_steps": step, "n_calls": n_calls,
                    "history": history_actions}

        history_actions.append(action)
        obs, _, dones, info = env.step([action])
        obs_text = obs[0]
        admissible = info.get("admissible_commands", [[]])[0]
        won = bool(info.get("won", [False])[0])
        done = bool(dones[0])

        if done or won:
            break

        # Append turn
        messages.append({"role": "assistant", "content": text})
        # Trim message history to last 6 to control context
        if len(messages) > 12:
            messages = [messages[0]] + messages[-8:]
        messages.append({
            "role": "user",
            "content": f"Observation:\n{obs_text}\n\nAdmissible commands:\n"
                       + "\n".join(f"- {c}" for c in admissible[:40])
                       + "\n\nWhat's your next action?",
        })

    return {"won": won, "n_steps": len(history_actions),
            "n_calls": n_calls, "history": history_actions,
            "error": None}


# --------------------------------------------------------------------------
# Multi-baseline driver
# --------------------------------------------------------------------------

def run_baseline(baseline_id: str, n_tasks: int, env_factory,
                 log_dir: str) -> Dict[str, Any]:
    """env_factory: () -> initialized AlfredTWEnv batch_size=1"""
    os.makedirs(log_dir, exist_ok=True)
    pred_path = os.path.join(log_dir, f"{baseline_id}_predictions.jsonl")
    fp = open(pred_path, "w")

    per_task = []
    t_start = time.time()
    env = env_factory()  # one env, reset per task
    for idx in range(n_tasks):
        t0 = time.time()
        _usage_log.clear()
        try:
            r = run_one_task(baseline_id, env, max_steps=30)
        except Exception as exc:
            traceback.print_exc()
            r = {"error": str(exc), "won": False, "n_steps": 0,
                 "n_calls": 0, "history": []}
        calls_used = list(_usage_log)
        usd_sum = sum(c["usd"] for c in calls_used)
        elapsed = time.time() - t0
        rec = {
            "task_idx": idx,
            "baseline": baseline_id,
            "task_success": bool(r["won"]),
            "n_steps": r["n_steps"],
            "n_llm_calls": r["n_calls"],
            "task_usd": usd_sum,
            "elapsed_s": elapsed,
            "error": r.get("error"),
        }
        fp.write(json.dumps(rec) + "\n"); fp.flush()
        per_task.append(rec)
        print(f"  [{baseline_id}] task={idx:>3} won={r['won']} "
              f"steps={r['n_steps']:>3} calls={r['n_calls']:>3} "
              f"usd={usd_sum:.4f} t={elapsed:.1f}s err={r.get('error') or '-'}")
    fp.close()
    total_elapsed = time.time() - t_start
    n = len(per_task)
    successes = sum(1 for r in per_task if r["task_success"])
    return {
        "baseline": baseline_id,
        "n_tasks": n,
        "task_success_count": successes,
        "task_success_rate_pp": round(100.0 * successes / max(1, n), 2),
        "harness_failure_count": sum(1 for r in per_task if r["error"]),
        "total_usd": sum(r["task_usd"] for r in per_task),
        "avg_usd_per_task": sum(r["task_usd"] for r in per_task) / max(1, n),
        "avg_steps": sum(r["n_steps"] for r in per_task) / max(1, n),
        "avg_calls_per_task": sum(r["n_llm_calls"] for r in per_task) / max(1, n),
        "avg_elapsed_s": sum(r["elapsed_s"] for r in per_task) / max(1, n),
        "total_elapsed_s": total_elapsed,
        "per_task": per_task,
    }


def make_env_factory(config_path: str):
    """Returns a factory that builds a fresh ALFWorld env (cached after 1st)."""
    config = yaml.safe_load(open(config_path))
    from alfworld.agents.environment import get_environment
    env_class = get_environment("AlfredTWEnv")
    env = env_class(config, train_eval="eval_out_of_distribution")
    env_inst = env.init_env(batch_size=1)

    def _factory():
        return env_inst
    return _factory


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/alfred_pilot.yaml")
    ap.add_argument("--baselines", nargs="+",
                    default=["B2_GlobalSeqWM"])
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--log_dir", default="runs/alfworld_pilot")
    ap.add_argument("--cell_cost_cap", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    env_factory = make_env_factory(args.config)

    all_results = {}
    cumulative_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        print(f"\n=== {baseline_id} on ALFWorld valid_unseen (n={args.n_tasks}) ===")
        try:
            r = run_baseline(baseline_id, args.n_tasks, env_factory, args.log_dir)
        except Exception as exc:
            print(f"!!! HARNESS FAILURE on {baseline_id}: {exc}")
            traceback.print_exc()
            all_results[baseline_id] = {"baseline": baseline_id,
                                         "error": str(exc),
                                         "task_success_rate_pp": None}
            continue
        all_results[baseline_id] = r
        cumulative_usd += r["total_usd"]
        print(f"  → TS={r['task_success_rate_pp']}pp, total=${r['total_usd']:.4f}, "
              f"cell_avg=${r['avg_usd_per_task']:.4f}/task")
        if r["total_usd"] > args.cell_cost_cap:
            print(f"!!! CELL > CAP {args.cell_cost_cap} — aborting Exp1 ALFWorld")
            break

    elapsed = time.time() - t_global
    summary = {
        "phase": "RUNNING_exp1_w2_d1",
        "benchmark": "alfworld_valid_unseen",
        "n_tasks_per_baseline": args.n_tasks,
        "seeds": 1,
        "model": MODEL_NAME,
        "elapsed_seconds": elapsed,
        "cumulative_usd": cumulative_usd,
        "per_baseline": all_results,
    }
    if "B8_FedCausalCompose" in all_results and "B2_GlobalSeqWM" in all_results:
        b8 = all_results["B8_FedCausalCompose"].get("task_success_rate_pp")
        b2 = all_results["B2_GlobalSeqWM"].get("task_success_rate_pp")
        if b8 is not None and b2 is not None:
            summary["G1_proxy_alfworld"] = {
                "B8_TS_pp": b8, "B2_TS_pp": b2,
                "B8_minus_B2_pp": round(b8 - b2, 2),
                "G1_5pp_pass": (b8 - b2) >= 5.0,
            }

    out_path = os.path.join(args.log_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(f"Total: ${cumulative_usd:.4f}, elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
