"""
P1.2: 3-seed ALF rerun for 5 baselines.

Multi-seed strategy: same 134 valid_unseen tasks, repeated 3 times.
Source of randomness: LLM agent at temperature=0.7 for seeds 1,2 (seed 0 uses temp 0.0).

Actually for cleanest stat: ALF env is deterministic; agent at temp=0 is
deterministic given context. The only seed dependence is via temperature.
We use seed 0 at temp=0 (matches Exp4 single-seed) + seeds 1,2 at temp=0.5.
"""

import argparse, json, os, re, sys, time, traceback, yaml
import numpy as np
from typing import Any, Dict, List

AZURE_API_KEY = os.environ.get("FED_CAUSAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
AZURE_API_BASE = os.environ.get("FED_CAUSAL_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
MODEL_NAME = os.environ.get("FED_CAUSAL_MODEL", "openai/gpt-5.4-mini")
DEPLOYMENT = MODEL_NAME.rsplit("/", 1)[-1]
PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00

import litellm
_original_completion = litellm.completion
_usage_log: List[Dict[str, Any]] = []
_current_seed = 0
_current_temp = 0.0


def _patched_completion(*args, **kwargs):
    if AZURE_API_KEY:
        kwargs.setdefault("api_key", AZURE_API_KEY)
    if AZURE_API_BASE:
        kwargs.setdefault("api_base", AZURE_API_BASE)
    if not kwargs.get("model", "").endswith(DEPLOYMENT):
        kwargs["model"] = MODEL_NAME
    kwargs.setdefault("custom_llm_provider", "openai")
    kwargs["temperature"] = _current_temp
    if _current_temp > 0:
        kwargs["seed"] = _current_seed
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
    "B2_GlobalSeqWM": (
        "WORLD MODEL: GLOBAL SEQUENCE. Track the global event sequence and "
        "predict next states from this sequence pattern."),
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
        "GROUND-TRUTH OPERATIONAL DEPENDENCIES (ALFWorld PDDL-derived oracle) "
        "across the 6 modules (navigation, container_access, "
        "object_manipulation, appliance, object_property, task_monitor):\n"
        + ANNOTATED_EDGES_ALF +
        "\nUse this verified dependency list to anticipate downstream module "
        "effects when executing each action."
    ),
}


SEARCH_HINTS = """SEARCH STRATEGY (important!):
- Items are usually on COUNTERTOPS, in DRAWERS, on SHELVES, on STOVEBURNERS, or already in the SINKBASIN.
- The FRIDGE typically contains cold items; check after.
- CABINETS often hold cookware not raw ingredients.
- After finding the item, plan: heat=microwave, cool=fridge, clean=sinkbasin.

CRITICAL: do NOT close a container after finding what you need is elsewhere — save steps."""

FEWSHOT_DEMO = """=== EXAMPLE 1: heat ===
Observation: You see countertop 1, fridge 1, microwave 1.
Your task: put a hot apple in countertop 1.
Thought: check countertops for apple.
ACTION: go to countertop 1

Observation: countertop 1 has apple 1.
ACTION: take apple 1 from countertop 1
Observation: pick up apple 1.
ACTION: go to microwave 1
Observation: microwave 1 closed.
ACTION: open microwave 1
Observation: open microwave 1.
ACTION: put apple 1 in/on microwave 1
Observation: put apple 1 in/on microwave 1.
ACTION: close microwave 1
Observation: close microwave 1.
ACTION: heat apple 1 with microwave 1
Observation: heat apple 1.
ACTION: open microwave 1
ACTION: take apple 1 from microwave 1
ACTION: go to countertop 1
ACTION: put apple 1 in/on countertop 1
Observation: TASK COMPLETE.
=== END ==="""

SYSTEM_INSTRUCTIONS = f"""You are an agent in ALFWorld. Use ReAct.
OUTPUT: Thought: <one sentence>\nACTION: <exact admissible command>
ACTION must match admissible. Max 50 steps.
{SEARCH_HINTS}
{FEWSHOT_DEMO}
Now solve the real task."""

ACTION_RE = re.compile(r"ACTION:\s*(.+?)$", re.IGNORECASE | re.MULTILINE)


def extract_action(text, admissible):
    m = ACTION_RE.search(text)
    if m:
        cmd = m.group(1).strip().rstrip(".").strip("'\"")
        if cmd in admissible: return cmd
        for adm in admissible:
            if cmd.lower() == adm.lower(): return adm
        for adm in admissible:
            if cmd.lower() in adm.lower() or adm.lower() in cmd.lower(): return adm
    for adm in admissible:
        if adm == "look": return adm
    return admissible[0] if admissible else "look"


def run_one_task(baseline_id, env, max_steps=50):
    header = BASELINE_HEADERS[baseline_id]
    obs, info = env.reset()
    obs_text = obs[0]
    admissible = info.get("admissible_commands", [[]])[0]
    won = False; n_calls = 0
    last_obs = ""; same_obs_count = 0

    system = f"{header}\n\n{SYSTEM_INSTRUCTIONS}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": format_user_msg(obs_text, admissible)}]

    for step in range(max_steps):
        try:
            res = litellm.completion(model=MODEL_NAME, messages=messages,
                                     max_tokens=128)
            n_calls += 1
            text = res.choices[0].message.content or ""
            action = extract_action(text, admissible)
        except Exception as exc:
            return {"error": str(exc), "won": False, "n_steps": step,
                    "n_calls": n_calls}
        obs, _, dones, info = env.step([action])
        obs_text = obs[0]
        admissible = info.get("admissible_commands", [[]])[0]
        won = bool(info.get("won", [False])[0])
        done = bool(dones[0])
        if done or won: break
        if obs_text.strip() == last_obs.strip(): same_obs_count += 1
        else: same_obs_count = 0
        last_obs = obs_text
        messages.append({"role": "assistant", "content": text})
        if len(messages) > 14: messages = [messages[0]] + messages[-12:]
        messages.append({"role": "user",
                         "content": format_user_msg(obs_text, admissible)})
        if same_obs_count >= 4:
            return {"won": False, "n_steps": step + 1, "n_calls": n_calls,
                    "error": "stuck"}
    return {"won": won, "n_steps": step + 1 if not (done or won) else step,
            "n_calls": n_calls, "error": None}


def format_user_msg(obs, admissible):
    adm_str = "\n".join(f"- {c}" for c in admissible[:40])
    return f"Observation: {obs}\n\nAdmissible:\n{adm_str}\n\nThought + ACTION:"


def make_env_factory(config_path):
    config = yaml.safe_load(open(config_path))
    from alfworld.agents.environment import get_environment
    env_class = get_environment("AlfredTWEnv")
    env = env_class(config, train_eval="eval_out_of_distribution")
    env_inst = env.init_env(batch_size=1)
    return lambda: env_inst


def wilson_ci_halfwidth(p_hat, n, z=1.96):
    if n == 0: return 0.0
    p = p_hat / 100.0
    denom = 1 + z**2 / n
    margin = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return margin * 100


def main():
    global _current_seed, _current_temp
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/alfred_pilot.yaml")
    ap.add_argument("--baselines", nargs="+",
                    default=["B2_GlobalSeqWM", "B7d_AnnotatedNoFraming_NoInt",
                             "B8d_AnnotatedNoFraming",
                             "B9_AnnotatedNoFramingNoControl",
                             "B10d_AnnotatedNoFraming_Oracle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_tasks", type=int, default=134)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--log_dir",
                    default="runs/p12_3seed_alf")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    env_factory = make_env_factory(args.config)

    all_results = {}
    cum_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        all_results[baseline_id] = []
        for seed in args.seeds:
            _current_seed = seed
            _current_temp = 0.0 if seed == 0 else 0.5
            print(f"\n=== {baseline_id} seed={seed} temp={_current_temp} ===", flush=True)
            env = env_factory()
            pred_path = os.path.join(args.log_dir, f"{baseline_id}_seed{seed}_predictions.jsonl")
            fp = open(pred_path, "w")
            per_task = []
            t_start = time.time()
            for idx in range(args.n_tasks):
                t0 = time.time()
                _usage_log.clear()
                try:
                    r = run_one_task(baseline_id, env, max_steps=args.max_steps)
                except Exception as exc:
                    traceback.print_exc()
                    r = {"error": str(exc), "won": False, "n_steps": 0, "n_calls": 0}
                usd_sum = sum(c["usd"] for c in _usage_log)
                elapsed = time.time() - t0
                rec = {"task_idx": idx, "baseline": baseline_id, "seed": seed,
                       "task_success": bool(r["won"]),
                       "n_steps": r["n_steps"], "n_llm_calls": r["n_calls"],
                       "task_usd": usd_sum, "elapsed_s": elapsed,
                       "error": r.get("error")}
                fp.write(json.dumps(rec) + "\n"); fp.flush()
                per_task.append(rec)
                if idx % 10 == 0:
                    print(f"  task={idx} won={r['won']} t={elapsed:.0f}s "
                          f"usd={usd_sum:.4f}", flush=True)
            fp.close()
            n = len(per_task); successes = sum(1 for r in per_task if r["task_success"])
            r_seed = {"baseline": baseline_id, "seed": seed, "n_tasks": n,
                      "task_success_count": successes,
                      "task_success_rate_pp": round(100.0 * successes / max(1, n), 2),
                      "total_usd": sum(r["task_usd"] for r in per_task),
                      "total_elapsed_s": time.time() - t_start}
            all_results[baseline_id].append(r_seed)
            cum_usd += r_seed["total_usd"]
            print(f"  → seed {seed}: TS={r_seed['task_success_rate_pp']}, "
                  f"usd=${r_seed['total_usd']:.4f}, cum=${cum_usd:.4f}", flush=True)

    elapsed = time.time() - t_global
    print("\n=== AGGREGATE ===", flush=True)
    print(f"{'Baseline':<35} {'s0':>8} {'s1':>8} {'s2':>8} {'mean':>8} {'CI':>8}")
    aggregate = {}
    for b, seed_runs in all_results.items():
        ts_seeds = [r["task_success_rate_pp"] for r in seed_runs]
        total_n = sum(r["n_tasks"] for r in seed_runs)
        total_won = sum(r["task_success_count"] for r in seed_runs)
        mean = 100.0 * total_won / max(1, total_n)
        ci = wilson_ci_halfwidth(mean, total_n)
        std = float(np.std(ts_seeds)) if len(ts_seeds) > 1 else 0.0
        aggregate[b] = {"ts_per_seed": ts_seeds, "mean_TS_pooled": round(mean, 2),
                        "total_n": total_n, "total_won": total_won,
                        "wilson_CI_halfwidth_pp": round(ci, 2),
                        "std_across_seeds_pp": round(std, 2)}
        print(f"{b:<35} {ts_seeds[0]:>8.2f} {ts_seeds[1]:>8.2f} {ts_seeds[2]:>8.2f} "
              f"{mean:>8.2f} {ci:>+8.2f}")

    summary = {"phase": "P1.2_3seed_alf",
               "baselines": args.baselines, "seeds": args.seeds,
               "n_tasks": args.n_tasks,
               "elapsed_seconds": elapsed, "cumulative_usd": cum_usd,
               "per_baseline_per_seed": all_results, "aggregate": aggregate}
    out = os.path.join(args.log_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}, elapsed {elapsed:.0f}s, cost ${cum_usd:.4f}")


if __name__ == "__main__":
    sys.exit(main())
