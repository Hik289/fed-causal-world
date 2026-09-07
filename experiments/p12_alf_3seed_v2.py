"""
exp1_alfworld_exp4.py — Exp4 (W2 D4-5) on ALF for missing baselines + B9.

Includes:
  - B0/B1/B2/B3/B4/B5 with their original prompts (no causal info) — full 134
  - B7d (A4 framing, obs-only annotation) — full 134
  - B9_AnnotatedNoFramingNoControl (B8d minus "anticipate downstream effects"
    suggestion = causal info present but no control hint)
  - B10d_AnnotatedNoFraming_Oracle (B10 with A4-style sanity)

Already done (don't re-run):
  - B8d full 134 (R1 result)

Note: B0/B1/B2/B3/B4/B5 don't carry causal info, so A4 framing doesn't apply
— we use the original v3 prompts. Their main-table value is unchanged from
the partial 48-58 data we collected earlier (we just extend to 134).
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

# API configuration is read from the environment.
AZURE_API_KEY = os.environ.get("FED_CAUSAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
AZURE_API_BASE = os.environ.get("FED_CAUSAL_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
MODEL_NAME = os.environ.get("FED_CAUSAL_MODEL", "openai/gpt-5.4-mini")
DEPLOYMENT = MODEL_NAME.rsplit("/", 1)[-1]
PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00

import litellm
_TEMP = 0.0
_SEED = 0
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


# ====================================================================
# ALF prompt headers (mix: original + A4-style)
# ====================================================================

# Same 10 annotated cross-module edges as R1
ANNOTATED_EDGES_ALF = (
    "navigation→container_access (agent_at unlocks visible_inside of receptacle), "
    "container_access→object_manipulation (container_opened enables pick of contents), "
    "navigation→object_manipulation (agent_at receptacle enables pick at that recep), "
    "object_manipulation→appliance (object_inserted_appliance enables effect on object), "
    "appliance→object_property (effect_applied flips object property after timer), "
    "appliance→object_property (sink/dishwasher effect cleans), "
    "object_manipulation→object_property (object_placed in sink → is_clean=1), "
    "object_property→task_monitor (property_changed satisfies goal predicate), "
    "object_manipulation→task_monitor (object_placed satisfies in(obj, dst) predicate), "
    "task_monitor→TERMINAL (goal_satisfied for all predicates → task_complete=1)."
)


BASELINE_HEADERS = {
    # ---- Non-causal baselines (no A4 needed; same prompts as v3) ----
    "B0_NoWM": "WORLD MODEL: None. Act greedily; do not plan ahead.",
    "B1_LocalWM": (
        "WORLD MODEL: Per-module LOCAL only. Track current module state only."),
    "B2_GlobalSeqWM": (
        "WORLD MODEL: GLOBAL SEQUENCE. Track the global event sequence and "
        "predict next states from this sequence pattern."),
    "B3_TransitionGraph": "WORLD MODEL: GLOBAL TRANSITION GRAPH.",
    "B4_CorrelationWM": "WORLD MODEL: CORRELATION-BASED.",
    "B5_TemporalWM": (
        "WORLD MODEL: TEMPORAL PRECEDENCE. Appliances toggled now will flip "
        "properties 1-3 steps later."),

    # ---- A4-style causal baselines ----
    "B7d_AnnotatedNoFraming_NoInt": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the ALFWorld "
        "household (navigation, container_access, object_manipulation, "
        "appliance, object_property, task_monitor), inferred from "
        "observational traces (not from controlled interventions):\n"
        + ANNOTATED_EDGES_ALF +
        "\nUse the dependency list above to anticipate downstream module "
        "effects when executing each action."
    ),
    "B8d_AnnotatedNoFraming": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the ALFWorld "
        "household (navigation, container_access, object_manipulation, "
        "appliance, object_property, task_monitor):\n"
        + ANNOTATED_EDGES_ALF +
        "\nUse the dependency list above to anticipate downstream module "
        "effects when executing each action (e.g. agent_at unlocks the "
        "container, opening unlocks pick, inserting into appliance + toggle "
        "flips property after timer)."
    ),
    "B9_AnnotatedNoFramingNoControl": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the ALFWorld "
        "household (navigation, container_access, object_manipulation, "
        "appliance, object_property, task_monitor):\n"
        + ANNOTATED_EDGES_ALF +
        "\nThese dependencies describe how the modules interact."
    ),
    "B10d_AnnotatedNoFraming_Oracle": (
        # B10 sanity: A4-style framing applied to oracle. Same as B8d here
        # because ALF modular_spec edges = oracle edges (10 deterministic
        # PDDL-derived). Difference vs B8d is purely epistemic anchor.
        "GROUND-TRUTH OPERATIONAL DEPENDENCIES (ALFWorld PDDL-derived oracle) "
        "across the 6 modules (navigation, container_access, "
        "object_manipulation, appliance, object_property, task_monitor):\n"
        + ANNOTATED_EDGES_ALF +
        "\nUse this verified dependency list to anticipate downstream module "
        "effects when executing each action."
    ),
}


SEARCH_HINTS = """SEARCH STRATEGY (important!):
- Items are usually on COUNTERTOPS, in DRAWERS, on SHELVES, on STOVEBURNERS,
  or already in the SINKBASIN. Search these FIRST.
- The FRIDGE typically contains cold/dairy items; check it only after.
- CABINETS often hold cookware (bowls, pots) not raw ingredients.
- After finding the item, plan: heat=microwave, cool=fridge, clean=sinkbasin,
  put-on-light=desklamp.

CRITICAL: do NOT close a container after finding what you need is elsewhere —
just walk away. Save steps.
"""

FEWSHOT_DEMO = """=== EXAMPLE 1: heat task ===
Observation: -= Welcome to TextWorld, ALFRED! =-
You see a cabinet 1, a countertop 1, a fridge 1, a microwave 1, a sinkbasin 1.
Your task is to: put a hot apple in countertop 1.
Thought: I need an apple. Check countertops first since fruit is often there.
ACTION: go to countertop 1
Observation: You arrive at countertop 1. On the countertop 1, you see a apple 1.
ACTION: take apple 1 from countertop 1
Observation: You pick up the apple 1.
ACTION: go to microwave 1
Observation: You arrive at microwave 1. The microwave 1 is closed.
ACTION: open microwave 1
Observation: You open the microwave 1.
ACTION: put apple 1 in/on microwave 1
Observation: You put the apple 1 in/on the microwave 1.
ACTION: close microwave 1
Observation: You close the microwave 1.
ACTION: heat apple 1 with microwave 1
Observation: You heat the apple 1 using the microwave 1.
ACTION: open microwave 1
Observation: In the microwave 1 you see a apple 1.
ACTION: take apple 1 from microwave 1
Observation: You pick up the apple 1.
ACTION: go to countertop 1
Observation: You arrive at countertop 1.
ACTION: put apple 1 in/on countertop 1
Observation: You put the apple 1 in/on the countertop 1. *** TASK COMPLETE ***
=== END EXAMPLE 1 ===

=== EXAMPLE 2: cool task ===
Observation: -= Welcome to TextWorld, ALFRED! =-
You see a cabinet 1, a countertop 1, a fridge 1, a microwave 1, a drawer 1.
Your task is to: put a cool mug in cabinet 1.
Thought: I need a mug. Check countertops and drawers (mugs often there).
ACTION: go to countertop 1
Observation: You arrive at countertop 1. On the countertop 1, you see a knife 1.
Thought: No mug here. Try drawer.
ACTION: go to drawer 1
Observation: You arrive at drawer 1. The drawer 1 is closed.
ACTION: open drawer 1
Observation: You open the drawer 1. In it, you see a mug 1.
ACTION: take mug 1 from drawer 1
Observation: You pick up the mug 1.
ACTION: go to fridge 1
Observation: You arrive at fridge 1. The fridge 1 is closed.
ACTION: open fridge 1
Observation: You open the fridge 1.
ACTION: put mug 1 in/on fridge 1
Observation: You put the mug 1 in/on the fridge 1.
ACTION: cool mug 1 with fridge 1
Observation: You cool the mug 1 using the fridge 1.
ACTION: take mug 1 from fridge 1
Observation: You pick up the mug 1.
ACTION: go to cabinet 1
Observation: You arrive at cabinet 1. The cabinet 1 is closed.
ACTION: open cabinet 1
Observation: You open the cabinet 1.
ACTION: put mug 1 in/on cabinet 1
Observation: You put the mug 1 in/on the cabinet 1. *** TASK COMPLETE ***
=== END EXAMPLE 2 ==="""


SYSTEM_INSTRUCTIONS = f"""You are an agent in an ALFWorld text environment.
Your task is described in the first observation. Each turn you see an
observation and a list of admissible commands. Use ReAct format.

OUTPUT FORMAT (mandatory):
Thought: <one short sentence>
ACTION: <exact command from admissible list>

CRITICAL RULES:
- ACTION must EXACTLY match an admissible command.
- Plan: search for item → take it → process it (heat/cool/clean) → place at destination.
- Maximum 50 steps. Be efficient.

{SEARCH_HINTS}

{FEWSHOT_DEMO}

Now solve the real task."""


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
        for adm in admissible:
            if cmd.lower() in adm.lower() or adm.lower() in cmd.lower():
                return adm
    for adm in admissible:
        if adm == "look":
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
    last_obs = ""
    same_obs_count = 0
    # Track state-regression-style stats
    n_redundant_actions = 0

    system = f"{header}\n\n{SYSTEM_INSTRUCTIONS}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _format_user_msg(obs_text, admissible)},
    ]

    for step in range(max_steps):
        try:
            res = litellm.completion(
                model=MODEL_NAME, messages=messages,
                max_tokens=128, temperature=_TEMP, seed=_SEED,
            )
            n_calls += 1
            text = res.choices[0].message.content or ""
            action = extract_action(text, admissible)
        except Exception as exc:
            return {"error": f"LLM err step {step}: {exc}", "won": False,
                    "n_steps": step, "n_calls": n_calls,
                    "history": history_actions,
                    "redundant_actions": n_redundant_actions}

        history_actions.append(action)
        obs, _, dones, info = env.step([action])
        obs_text = obs[0]
        admissible = info.get("admissible_commands", [[]])[0]
        won = bool(info.get("won", [False])[0])
        done = bool(dones[0])

        if done or won:
            break

        if obs_text.strip() == last_obs.strip():
            same_obs_count += 1
            n_redundant_actions += 1
        else:
            same_obs_count = 0
        last_obs = obs_text

        messages.append({"role": "assistant", "content": text})
        if len(messages) > 14:
            messages = [messages[0]] + messages[-12:]
        messages.append({"role": "user",
                         "content": _format_user_msg(obs_text, admissible)})

        if same_obs_count >= 4:
            return {"won": False, "n_steps": len(history_actions),
                    "n_calls": n_calls, "history": history_actions,
                    "redundant_actions": n_redundant_actions,
                    "error": "stuck_4_consecutive"}

    return {"won": won, "n_steps": len(history_actions),
            "n_calls": n_calls, "history": history_actions,
            "redundant_actions": n_redundant_actions,
            "error": None}


def _format_user_msg(obs: str, admissible: List[str]) -> str:
    adm_str = "\n".join(f"- {c}" for c in admissible[:40])
    if len(admissible) > 40:
        adm_str += f"\n- ... ({len(admissible) - 40} more)"
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
                 "n_calls": 0, "history": [], "redundant_actions": 0}
        usd_sum = sum(c["usd"] for c in _usage_log)
        elapsed = time.time() - t0
        rec = {
            "task_idx": idx, "baseline": baseline_id,
            "task_success": bool(r["won"]),
            "n_steps": r["n_steps"], "n_llm_calls": r["n_calls"],
            "redundant_actions": r.get("redundant_actions", 0),
            "task_usd": usd_sum, "elapsed_s": elapsed,
            "error": r.get("error"),
        }
        fp.write(json.dumps(rec) + "\n"); fp.flush()
        per_task.append(rec)
        # Touch heartbeat every 5 tasks to avoid 30min lapse
        if idx % 5 == 0:
            try:
                with open("/tmp/fedcausalworld_heartbeat_alf", "w") as pf:
                    pf.write("")
            except Exception:
                pass
        print(f"  [{baseline_id}] task={idx:>3} won={r['won']} "
              f"steps={r['n_steps']:>3} calls={r['n_calls']:>3} "
              f"red={rec['redundant_actions']:>2} usd={usd_sum:.4f} "
              f"t={elapsed:.1f}s err={r.get('error') or '-'}", flush=True)
    fp.close()
    total_elapsed = time.time() - t_start
    n = len(per_task)
    successes = sum(1 for r in per_task if r["task_success"])
    state_reg_total = sum(r["redundant_actions"] for r in per_task)
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
        "state_regression_total": state_reg_total,
        "state_regression_avg": state_reg_total / max(1, n),
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
    ap.add_argument("--config", default="configs/alfred_pilot.yaml")
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--n_tasks", type=int, default=134)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--log_dir", required=True)
    args = ap.parse_args()
    global _TEMP, _SEED
    _TEMP = args.temperature
    _SEED = args.seed
    os.makedirs(args.log_dir, exist_ok=True)
    env_factory = make_env_factory(args.config)

    all_results = {}
    cumulative_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        if baseline_id not in BASELINE_HEADERS:
            print(f"!!! unknown baseline {baseline_id}", flush=True)
            continue
        print(f"\n=== {baseline_id} on ALF v_unseen (n={args.n_tasks}) ===", flush=True)
        r = run_baseline(baseline_id, args.n_tasks, env_factory,
                         args.log_dir, max_steps=args.max_steps)
        all_results[baseline_id] = r
        cumulative_usd += r["total_usd"]
        print(f"  → TS={r['task_success_rate_pp']}pp, "
              f"total=${r['total_usd']:.4f}, "
              f"state_reg_avg={r['state_regression_avg']:.1f}", flush=True)

    elapsed = time.time() - t_global
    summary = {
        "phase": "RUNNING_exp4_w2_d4_d5",
        "benchmark": "alfworld_valid_unseen",
        "n_tasks_per_baseline": args.n_tasks,
        "max_steps": args.max_steps,
        "seeds": 1, "model": MODEL_NAME,
        "elapsed_seconds": elapsed,
        "cumulative_usd": cumulative_usd,
        "per_baseline": all_results,
    }
    out_path = os.path.join(args.log_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(f"Total: ${cumulative_usd:.4f}, elapsed {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
