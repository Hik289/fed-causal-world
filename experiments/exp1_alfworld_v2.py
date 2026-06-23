"""
exp1_alfworld_v2.py — ALFWorld X fix per Launcher directive (2026-06-21 05:50 UTC).

Changes vs v1:
  1. max_steps: 30 → 50 (ALF valid_unseen mean traj = 42 steps; need buffer)
  2. ReAct-style few-shot demo (1 complete task from ALFRED PDDL-derived oracle)
  3. Clearer system prompt: explicit Thought / Action pattern
  4. Trim observation context but keep last-3 turns
  5. Admissible-command list still given, but model is instructed to reason first
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
import traceback
import yaml
from typing import Any, Dict, List

AZURE_API_KEY = "YOUR_AZURE_API_KEY"
AZURE_API_BASE = "YOUR_AZURE_ENDPOINT"
MODEL_NAME = "openai/gpt-5.4-mini"
DEPLOYMENT = "gpt-5.4-mini"
PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00

import litellm
_original_completion = litellm.completion
_usage_log: List[Dict[str, Any]] = []


def _patched_completion(*args, **kwargs):
    kwargs["api_key"] = AZURE_API_KEY
    kwargs["api_base"] = AZURE_API_BASE
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


# Modular summary identical to v1
MODULARIZATION_SUMMARY = (
    "ALFWorld has 6 functional modules: navigation, container_access, "
    "object_manipulation, appliance, object_property, task_monitor. "
    "Cross-module dependencies: navigation→container_access, "
    "container_access→object_manipulation, navigation→object_manipulation, "
    "object_manipulation→appliance, appliance→object_property, "
    "object_manipulation→object_property, object_property→task_monitor, "
    "object_manipulation→task_monitor."
)

ORACLE_EDGES = (
    "GROUND-TRUTH causal edges: navigation→container_access (agent_at unlocks "
    "visible_inside), container_access→object_manipulation (container_opened "
    "enables pick), navigation→object_manipulation (agent_at enables pick), "
    "object_manipulation→appliance (insert enables effect), "
    "appliance→object_property (effect_applied flips property after timer), "
    "object_manipulation→object_property (place in sink → is_clean=1), "
    "object_property→task_monitor (property_changed satisfies predicate), "
    "object_manipulation→task_monitor (place satisfies in() predicate)."
)


BASELINE_HEADERS = {
    "B0_NoWM": "WORLD MODEL: None. Act greedily; do not plan ahead.",
    "B1_LocalWM": (
        "WORLD MODEL: Per-module LOCAL only. Track only the current "
        "module state (where you are OR what's in your hand OR what's open). "
        "Do not chain cross-module effects."),
    "B2_GlobalSeqWM": (
        "WORLD MODEL: GLOBAL SEQUENCE. Track the global event sequence "
        "(navigation, container open/close, pick/place, appliance toggle, "
        "property changes). Predict next states from this sequence pattern."),
    "B3_TransitionGraph": (
        "WORLD MODEL: GLOBAL TRANSITION GRAPH. Maintain (state, action) → "
        "next_state transitions."),
    "B4_CorrelationWM": (
        "WORLD MODEL: CORRELATION-BASED. Track which events co-occur."),
    "B5_TemporalWM": (
        "WORLD MODEL: TEMPORAL PRECEDENCE. Track event order and lag. "
        "Appliances toggled now will flip properties 1-3 steps later."),
    "B7_CausalWMNoInt": (
        "WORLD MODEL: CAUSAL graph from OBSERVATIONAL data only. " +
        MODULARIZATION_SUMMARY),
    "B8_FedCausalCompose": (
        "WORLD MODEL: FedCausalCompose — full causal graph + control. " +
        MODULARIZATION_SUMMARY +
        " For each action: INSPECT upstream prereqs (e.g. agent_at receptacle "
        "before pick), PREDICT downstream effects (e.g. toggle microwave → "
        "object heats next turn), BLOCK actions violating these constraints."),
    "B10_OracleCausalWM": (
        "WORLD MODEL: ORACLE causal graph. " + ORACLE_EDGES),
}


# Standard ReAct ALFWorld demo (heat task — covers nav + container +
# manip + appliance + property + monitor). Adapted from Yao 2022 ReAct paper.
FEWSHOT_DEMO = """=== EXAMPLE TASK (for format reference) ===
Observation: -= Welcome to TextWorld, ALFRED! =-
You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a fridge 1, a microwave 1, a stoveburner 1, and a sinkbasin 1.
Your task is to: put a hot tomato in countertop 1.

Thought: I need to find a tomato, heat it in microwave, then place on countertop. First find tomato.
ACTION: go to fridge 1

Observation: You arrive at fridge 1. The fridge 1 is closed.
Thought: Open fridge to find tomato.
ACTION: open fridge 1

Observation: You open the fridge 1. In it, you see a tomato 1, a egg 1, a bread 1.
Thought: Take the tomato.
ACTION: take tomato 1 from fridge 1

Observation: You pick up the tomato 1.
Thought: Now go to microwave to heat it.
ACTION: go to microwave 1

Observation: You arrive at microwave 1. The microwave 1 is closed.
Thought: Open microwave, insert tomato, close, then heat.
ACTION: open microwave 1

Observation: You open the microwave 1. The microwave 1 is empty.
ACTION: put tomato 1 in/on microwave 1

Observation: You put the tomato 1 in/on the microwave 1.
ACTION: close microwave 1

Observation: You close the microwave 1.
ACTION: heat tomato 1 with microwave 1

Observation: You heat the tomato 1 using the microwave 1.
Thought: Now retrieve the hot tomato and place on countertop.
ACTION: open microwave 1

Observation: You open the microwave 1. In it, you see a tomato 1.
ACTION: take tomato 1 from microwave 1

Observation: You pick up the tomato 1.
ACTION: go to countertop 1

Observation: You arrive at countertop 1. On it, you see a knife 1.
ACTION: put tomato 1 in/on countertop 1

Observation: You put the tomato 1 in/on the countertop 1. *** TASK COMPLETE ***
=== END EXAMPLE ==="""


SYSTEM_INSTRUCTIONS = f"""You are an agent in an ALFWorld text environment.
Your task is described in the first observation. You will receive an
observation and a list of admissible commands each turn. Use ReAct: think
briefly, then output an action.

OUTPUT FORMAT (mandatory):
Thought: <one short sentence reasoning about next step>
ACTION: <exact command from the admissible commands list>

CRITICAL RULES:
- ACTION must EXACTLY match an item in the admissible commands list.
- Do NOT invent or hallucinate commands.
- Plan multi-step: to heat → go to microwave → open → put object → close → heat.
- To clean → go to sink → put object in sink → toggle sink.
- To cool → go to fridge → open → put object → close.
- After completing a sub-goal, immediately move to the next.
- Maximum 50 steps total. Be efficient.

{FEWSHOT_DEMO}

Now solve the real task below."""


ACTION_RE = re.compile(r"ACTION:\s*(.+?)$", re.IGNORECASE | re.MULTILINE)


def extract_action(text: str, admissible: List[str]) -> str:
    m = ACTION_RE.search(text)
    if m:
        cmd = m.group(1).strip().rstrip(".").strip("'\"")
        if cmd in admissible:
            return cmd
        for adm in admissible:
            if cmd.lower() == adm.lower():
                return adm
        # Substring match
        for adm in admissible:
            if cmd.lower() in adm.lower() or adm.lower() in cmd.lower():
                return adm
    # Fallback: any 'look' if available, else first
    for adm in admissible:
        if adm.startswith("look"):
            return adm
    return admissible[0] if admissible else "look"


def run_one_task(baseline_id: str, env, max_steps: int = 50) -> Dict[str, Any]:
    header = BASELINE_HEADERS[baseline_id]
    obs, info = env.reset()
    obs_text = obs[0]
    admissible = info.get("admissible_commands", [[]])[0]
    won = False
    n_calls = 0
    history_actions = []
    failed_actions = 0   # actions that don't change the world

    system = f"{header}\n\n{SYSTEM_INSTRUCTIONS}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _format_user_msg(obs_text, admissible)},
    ]

    last_obs = ""
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
            return {"error": f"LLM err step {step}: {exc}", "won": False,
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

        # Track stuck-in-loop
        if obs_text.strip() == last_obs.strip():
            failed_actions += 1
        else:
            failed_actions = 0
        last_obs = obs_text

        # Append turn (keep messages compact)
        messages.append({"role": "assistant", "content": text})
        if len(messages) > 14:
            messages = [messages[0]] + messages[-12:]
        messages.append({
            "role": "user",
            "content": _format_user_msg(obs_text, admissible),
        })

        # If stuck >5 consecutive failed actions, terminate (saves API)
        if failed_actions >= 5:
            return {"won": False, "n_steps": len(history_actions),
                    "n_calls": n_calls, "history": history_actions,
                    "error": "stuck_5_consecutive_no_change"}

    return {"won": won, "n_steps": len(history_actions),
            "n_calls": n_calls, "history": history_actions,
            "error": None}


def _format_user_msg(obs: str, admissible: List[str]) -> str:
    # Truncate admissible to first 30 (most are go-to-receptacle filler)
    adm_str = "\n".join(f"- {c}" for c in admissible[:30])
    if len(admissible) > 30:
        adm_str += f"\n- ... ({len(admissible) - 30} more not shown)"
    return f"Observation: {obs}\n\nAdmissible commands:\n{adm_str}\n\nThought + ACTION:"


def run_baseline(baseline_id: str, n_tasks: int, env_factory,
                 log_dir: str, max_steps: int = 50) -> Dict[str, Any]:
    os.makedirs(log_dir, exist_ok=True)
    pred_path = os.path.join(log_dir, f"{baseline_id}_predictions.jsonl")
    fp = open(pred_path, "w")

    per_task = []
    t_start = time.time()
    env = env_factory()
    for idx in range(n_tasks):
        t0 = time.time()
        _usage_log.clear()
        try:
            r = run_one_task(baseline_id, env, max_steps=max_steps)
        except Exception as exc:
            traceback.print_exc()
            r = {"error": str(exc), "won": False, "n_steps": 0,
                 "n_calls": 0, "history": []}
        calls_used = list(_usage_log)
        usd_sum = sum(c["usd"] for c in calls_used)
        elapsed = time.time() - t0
        rec = {
            "task_idx": idx, "baseline": baseline_id,
            "task_success": bool(r["won"]),
            "n_steps": r["n_steps"], "n_llm_calls": r["n_calls"],
            "task_usd": usd_sum, "elapsed_s": elapsed,
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
        "baseline": baseline_id, "n_tasks": n,
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
    config = yaml.safe_load(open(config_path))
    from alfworld.agents.environment import get_environment
    env_class = get_environment("AlfredTWEnv")
    env = env_class(config, train_eval="eval_out_of_distribution")
    env_inst = env.init_env(batch_size=1)
    def _factory(): return env_inst
    return _factory


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/home/user/fedcausalworld/experiments/exp1_run/alf_config/alfred_pilot.yaml")
    ap.add_argument("--baselines", nargs="+", default=["B2_GlobalSeqWM"])
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--log_dir", default="/home/user/fedcausalworld/experiments/exp1_run/alfworld_pilot_v2")
    ap.add_argument("--cell_cost_cap", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    env_factory = make_env_factory(args.config)

    all_results = {}
    cumulative_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        print(f"\n=== {baseline_id} on ALFWorld v_unseen (n={args.n_tasks}, max_steps={args.max_steps}) ===")
        try:
            r = run_baseline(baseline_id, args.n_tasks, env_factory,
                             args.log_dir, max_steps=args.max_steps)
        except Exception as exc:
            print(f"!!! HARNESS FAILURE on {baseline_id}: {exc}")
            traceback.print_exc()
            all_results[baseline_id] = {"baseline": baseline_id, "error": str(exc)}
            continue
        all_results[baseline_id] = r
        cumulative_usd += r["total_usd"]
        print(f"  → TS={r['task_success_rate_pp']}pp, total=${r['total_usd']:.4f}, "
              f"cell_avg=${r['avg_usd_per_task']:.4f}/task")
        if r["total_usd"] > args.cell_cost_cap:
            print(f"!!! CELL > CAP {args.cell_cost_cap} — aborting")
            break

    elapsed = time.time() - t_global
    summary = {
        "phase": "RUNNING_exp1_alf_fix_x",
        "benchmark": "alfworld_valid_unseen",
        "n_tasks_per_baseline": args.n_tasks, "max_steps": args.max_steps,
        "seeds": 1, "model": MODEL_NAME,
        "elapsed_seconds": elapsed, "cumulative_usd": cumulative_usd,
        "per_baseline": all_results,
    }
    if "B8_FedCausalCompose" in all_results and "B2_GlobalSeqWM" in all_results:
        b8 = all_results["B8_FedCausalCompose"].get("task_success_rate_pp")
        b2 = all_results["B2_GlobalSeqWM"].get("task_success_rate_pp")
        if b8 is not None and b2 is not None:
            summary["G1_proxy_alfworld"] = {
                "B8_TS_pp": b8, "B2_TS_pp": b2,
                "B8_minus_B2_pp": round(b8 - b2, 2),
            }
    out_path = os.path.join(args.log_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(f"Total: ${cumulative_usd:.4f}, elapsed {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
