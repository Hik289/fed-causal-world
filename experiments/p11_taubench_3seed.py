"""
P1.1: 3-seed τ-bench rerun for 5 baselines (B2, B7d, B8d, B9, B10).

Multi-seed strategy: same 21 retail test tasks, repeat 3 times.
Sources of randomness:
  - User simulator LLM responses (Azure GPT-5.4-mini, temperature=0.7 instead
    of 0.0 to ensure variation across reruns — this is the standard
    multi-seed setup for LLM-agent benchmarks)
  - Tool-calling agent temperature kept at 0.0 (deterministic decisions
    given context)

Output: per-seed task-level Task Success, aggregated to mean ± Wilson CI.
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

# === API CREDS (set via env vars or fill below) ===
AZURE_API_KEY = "YOUR_AZURE_API_KEY"
AZURE_API_BASE = "YOUR_AZURE_ENDPOINT"
MODEL_NAME = "openai/gpt-5.4-mini"
DEPLOYMENT = "gpt-5.4-mini"
PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00

import litellm
_original_completion = litellm.completion
_usage_log: List[Dict[str, Any]] = []
_current_seed = 0  # mutated per-task by main()


def _patched_completion(*args, **kwargs):
    kwargs["api_key"] = AZURE_API_KEY
    kwargs["api_base"] = AZURE_API_BASE
    if not kwargs.get("model", "").endswith(DEPLOYMENT):
        kwargs["model"] = MODEL_NAME
    kwargs.setdefault("custom_llm_provider", "openai")
    # Apply seed-based randomness to user simulator only
    # (user simulator gets temperature 0.7; agent stays 0.0)
    # We identify user simulator by checking if "user" appears in system message
    msgs = kwargs.get("messages", [])
    sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
    if "user interacting" in sys_msg.lower() or "simulate the user" in sys_msg.lower():
        kwargs["temperature"] = 0.7
        kwargs["seed"] = _current_seed  # OpenAI-compatible seed param
    else:
        kwargs["temperature"] = 0.0
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


# Reuse prompt headers from existing scripts
ANNOTATED_EDGES_RETAIL = (
    "account→order (authentication gates create_order), "
    "order→payment (order_placed triggers authorize_payment), "
    "order→inventory (order_placed triggers reserve_stock), "
    "inventory→order (inventory_reserved drives order_status pending→draft), "
    "payment→order (payment_captured drives order_status pending→confirmed), "
    "order→shipment (order_confirmed triggers create_label), "
    "inventory+payment→shipment (AND mediator: both reserved AND captured), "
    "shipment→order (shipment_completed drives order_status shipped→delivered), "
    "shipment→refund (delayed refund_eligibility after return window), "
    "refund→payment (refund_issued triggers refund_payment), "
    "payment→order (payment_refunded drives order_status delivered→returned), "
    "order→inventory (order_cancelled releases reservation), "
    "order→payment (order_cancelled voids auth), "
    "account→shipment (address modulates delivery_days)."
)

# Oracle annotated edges (B10)
ORACLE_EDGES_FULL = (
    "GROUND-TRUTH causal interface edges (Oracle): "
    + ANNOTATED_EDGES_RETAIL +
    " Use this oracle graph to perform causal rollout and downstream-effect "
    "prediction with maximal confidence."
)

BASELINE_HEADERS = {
    "B2_GlobalSeqWM": (
        "WORLD MODEL: GLOBAL SEQUENCE. You track the global event sequence "
        "across all modules (account, order, payment, inventory, shipment, "
        "refund) and predict next states from this sequence. No explicit "
        "causal structure — purely sequential pattern matching."
    ),
    "B7d_AnnotatedNoFraming_NoInt": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the retail system "
        "(account, order, payment, inventory, shipment, refund), inferred "
        "from observational traces (not from controlled interventions):\n"
        + ANNOTATED_EDGES_RETAIL +
        "\nUse the dependency list above to anticipate downstream module "
        "effects when executing each tool call."
    ),
    "B8d_AnnotatedNoFraming": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the retail system "
        "(account, order, payment, inventory, shipment, refund):\n"
        + ANNOTATED_EDGES_RETAIL +
        "\nUse the dependency list above to anticipate downstream module "
        "effects when executing each tool call."
    ),
    "B9_AnnotatedNoFramingNoControl": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the retail system "
        "(account, order, payment, inventory, shipment, refund):\n"
        + ANNOTATED_EDGES_RETAIL +
        "\nThese dependencies describe how the modules interact."
    ),
    "B10_OracleCausalWM": (
        "WORLD MODEL: ORACLE causal graph (ground-truth). " + ORACLE_EDGES_FULL
    ),
}


from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.user import UserStrategy


class PromptHeaderAgent(ToolCallingAgent):
    def __init__(self, header: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wiki = f"{header}\n\n---\n\n{self.wiki}"


def run_baseline_seed(baseline_id: str, seed: int, task_indices: List[int],
                      log_dir: str) -> Dict[str, Any]:
    global _current_seed
    _current_seed = seed
    os.makedirs(log_dir, exist_ok=True)
    pred_path = os.path.join(log_dir, f"{baseline_id}_seed{seed}_predictions.jsonl")
    fp = open(pred_path, "w")
    header = BASELINE_HEADERS[baseline_id]
    per_task = []
    t_start = time.time()
    for idx in task_indices:
        t0 = time.time()
        _usage_log.clear()
        try:
            env = MockRetailDomainEnv(
                user_strategy=UserStrategy.LLM,
                user_model=MODEL_NAME, user_provider="openai",
                task_split="test", task_index=idx,
            )
            agent = PromptHeaderAgent(
                header=header,
                tools_info=env.tools_info, wiki=env.wiki,
                model=MODEL_NAME, provider="openai", temperature=0.0,
            )
            result = agent.solve(env, task_index=idx, max_num_steps=30)
            reward = float(result.reward)
            err = None
        except Exception as exc:
            reward = 0.0
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        elapsed = time.time() - t0
        usd_sum = sum(c["usd"] for c in _usage_log)
        rec = {
            "task_idx": idx, "baseline": baseline_id, "seed": seed,
            "reward": reward, "task_success": bool(reward >= 0.5),
            "n_llm_calls": len(_usage_log), "task_usd": usd_sum,
            "elapsed_s": elapsed, "error": err,
        }
        fp.write(json.dumps(rec) + "\n"); fp.flush()
        per_task.append(rec)
        print(f"  [{baseline_id} seed={seed}] task={idx:>3} rwd={reward:.2f} "
              f"calls={rec['n_llm_calls']:>2} usd={usd_sum:.4f} "
              f"t={elapsed:.1f}s err={err or '-'}", flush=True)
    fp.close()
    n = len(per_task)
    successes = sum(1 for r in per_task if r["task_success"])
    return {
        "baseline": baseline_id, "seed": seed,
        "n_tasks": n, "task_success_count": successes,
        "task_success_rate_pp": round(100.0 * successes / max(1, n), 2),
        "total_usd": sum(r["task_usd"] for r in per_task),
        "avg_calls_per_task": sum(r["n_llm_calls"] for r in per_task) / max(1, n),
        "total_elapsed_s": time.time() - t_start,
        "per_task": per_task,
    }


def wilson_ci_halfwidth(p_hat: float, n: int, z: float = 1.96) -> float:
    """Wilson 95% CI half-width for proportion p_hat with n trials."""
    if n == 0: return 0.0
    p = p_hat / 100.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return margin * 100  # back to pp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    default=["B2_GlobalSeqWM", "B7d_AnnotatedNoFraming_NoInt",
                             "B8d_AnnotatedNoFraming",
                             "B9_AnnotatedNoFramingNoControl",
                             "B10_OracleCausalWM"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--task_indices", type=int, nargs="+",
                    default=list(range(21)))
    ap.add_argument("--log_dir", default="/home/user/fedcausalworld/experiments/exp1_run/p11_3seed_tau")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    all_results = {}
    cum_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        all_results[baseline_id] = []
        for seed in args.seeds:
            print(f"\n=== {baseline_id} seed={seed} on τ-bench retail ===", flush=True)
            r = run_baseline_seed(baseline_id, seed, args.task_indices, args.log_dir)
            all_results[baseline_id].append(r)
            cum_usd += r["total_usd"]
            print(f"  → TS={r['task_success_rate_pp']}pp, cost=${r['total_usd']:.4f}, "
                  f"cum=${cum_usd:.4f}", flush=True)

    elapsed = time.time() - t_global

    # Aggregate per baseline (mean across seeds + Wilson CI on combined data)
    print("\n=== AGGREGATE ===")
    print(f"{'Baseline':<35} {'seed0_TS':>10} {'seed1_TS':>10} {'seed2_TS':>10} {'mean':>8} {'CI_half':>8}")
    aggregate = {}
    for b, seed_runs in all_results.items():
        ts_per_seed = [r["task_success_rate_pp"] for r in seed_runs]
        # Combine all tasks across seeds → pool of n_tasks × n_seeds for Wilson
        total_n = sum(r["n_tasks"] for r in seed_runs)
        total_won = sum(r["task_success_count"] for r in seed_runs)
        mean_ts = 100.0 * total_won / max(1, total_n)
        ci = wilson_ci_halfwidth(mean_ts, total_n)
        std_across_seeds = float(np.std(ts_per_seed)) if len(ts_per_seed) > 1 else 0.0
        aggregate[b] = {
            "ts_per_seed": ts_per_seed,
            "mean_TS_pooled": round(mean_ts, 2),
            "total_n": total_n,
            "total_won": total_won,
            "wilson_CI_halfwidth_pp": round(ci, 2),
            "std_across_seeds_pp": round(std_across_seeds, 2),
        }
        print(f"{b:<35} {ts_per_seed[0]:>10.2f} {ts_per_seed[1]:>10.2f} "
              f"{ts_per_seed[2]:>10.2f} {mean_ts:>8.2f} {ci:>+8.2f}")

    # G1 / G4: B8d vs B2 paired analysis
    if "B8d_AnnotatedNoFraming" in aggregate and "B2_GlobalSeqWM" in aggregate:
        b8d = aggregate["B8d_AnnotatedNoFraming"]
        b2 = aggregate["B2_GlobalSeqWM"]
        gap = b8d["mean_TS_pooled"] - b2["mean_TS_pooled"]
        print(f"\nG1 proxy τ-bench (3-seed pooled n={b8d['total_n']}):")
        print(f"  B8d {b8d['mean_TS_pooled']}% ± {b8d['wilson_CI_halfwidth_pp']}pp")
        print(f"  B2  {b2['mean_TS_pooled']}% ± {b2['wilson_CI_halfwidth_pp']}pp")
        print(f"  Gap = {gap:+.2f}pp (gate ≥+5pp: {'PASS' if gap >= 5 else 'BORDERLINE/FAIL'})")
        # Paired McNemar-style: per-task agreement
        # combine all per-task across seeds, paired by (task_idx, seed)
        b8d_tasks = {(r['task_idx'], r['seed']): r['task_success']
                     for run in all_results["B8d_AnnotatedNoFraming"] for r in run['per_task']}
        b2_tasks = {(r['task_idx'], r['seed']): r['task_success']
                    for run in all_results["B2_GlobalSeqWM"] for r in run['per_task']}
        b8d_wins_b2_loses = sum(1 for k in b8d_tasks if b8d_tasks[k] and not b2_tasks.get(k, False))
        b2_wins_b8d_loses = sum(1 for k in b2_tasks if b2_tasks[k] and not b8d_tasks.get(k, False))
        print(f"  Paired count: B8d wins B2 loses = {b8d_wins_b2_loses}, "
              f"B2 wins B8d loses = {b2_wins_b8d_loses}")

    summary = {
        "phase": "P1.1_3seed_taubench",
        "baselines": args.baselines, "seeds": args.seeds,
        "task_indices": args.task_indices,
        "elapsed_seconds": elapsed,
        "cumulative_usd": cum_usd,
        "per_baseline_per_seed": all_results,
        "aggregate": aggregate,
    }
    out = os.path.join(args.log_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}, elapsed {elapsed:.0f}s, total cost ${cum_usd:.4f}")


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
