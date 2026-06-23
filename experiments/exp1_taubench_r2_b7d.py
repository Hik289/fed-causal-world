"""
exp1_taubench_r2_b7d.py — R2 ablation: B7d = B7 with A4-style framing.

Per specification (2026-06-21 11:18 UTC): test if applying A4 framing
(annotated edges + drop "confounded/ambiguous/causal" anchor) also rescues B7.

B7 main was: "WORLD MODEL: CAUSAL graph from OBSERVATIONAL data only" +
"learned from observation only (NO intervention-based validation)" +
"some edges may be confounded or directionally ambiguous"
→ TS = 23.81% on τ-bench (main result)

B7d (R2): drop all "causal/validated/confounded/ambiguous" framing.  Use
B10-like "OPERATIONAL DEPENDENCIES" header + same annotated edge list as A4.
The ONLY difference vs A4 B8d is a semantic flag: "observational-only,
no intervention evidence" — i.e. B7 still has weaker info than B8.

For isolation purposes we keep the spec-level distinction but drop the
epistemic-hedging wording that may have hurt B7.
"""

from __future__ import annotations
import json
import os
import sys
import time
import traceback
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


# Same 14 annotated edges as A4 (B8d).
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


# B7d differs from B8d only in a brief tag about info source
# (observation-only vs intervention-validated).  Same operational dependency
# list, same B10-style framing.  Critically, drop "confounded / ambiguous /
# causal" anchor wording that B7 main had.
BASELINE_HEADERS_R2 = {
    "B7d_AnnotatedNoFraming_NoInt": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the retail system "
        "(account, order, payment, inventory, shipment, refund), inferred "
        "from observational traces (not from controlled interventions):\n"
        + ANNOTATED_EDGES_RETAIL +
        "\nUse the dependency list above to anticipate downstream module "
        "effects when executing each tool call."
    ),
}


from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.user import UserStrategy


class PromptHeaderAgent(ToolCallingAgent):
    def __init__(self, header: str, *args, **kwargs):
        self._baseline_header = header
        super().__init__(*args, **kwargs)
        self.wiki = f"{header}\n\n---\n\n{self.wiki}"


def run_one_baseline(baseline_id: str, task_indices: List[int],
                     log_dir: str) -> Dict[str, Any]:
    os.makedirs(log_dir, exist_ok=True)
    pred_path = os.path.join(log_dir, f"{baseline_id}_predictions.jsonl")
    fp = open(pred_path, "w")
    header = BASELINE_HEADERS_R2[baseline_id]
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
            n_msgs = len(result.messages)
            total_cost = float(result.total_cost or 0.0)
            err = None
        except Exception as exc:
            reward = 0.0; n_msgs = 0; total_cost = 0.0
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        elapsed = time.time() - t0
        usd_sum = sum(c["usd"] for c in _usage_log)
        rec = {
            "task_idx": idx, "baseline": baseline_id,
            "reward": reward, "task_success": bool(reward >= 0.5),
            "n_messages": n_msgs, "n_llm_calls": len(_usage_log),
            "task_usd": usd_sum, "tau_bench_reported_cost": total_cost,
            "elapsed_s": elapsed, "error": err,
        }
        fp.write(json.dumps(rec) + "\n"); fp.flush()
        per_task.append(rec)
        print(f"  [{baseline_id}] task={idx:>3} reward={reward:.2f} "
              f"calls={rec['n_llm_calls']:>3} usd={usd_sum:.4f} "
              f"t={elapsed:.1f}s err={err or '-'}", flush=True)
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
        "avg_calls_per_task": sum(r["n_llm_calls"] for r in per_task) / max(1, n),
        "avg_elapsed_s": sum(r["elapsed_s"] for r in per_task) / max(1, n),
        "total_elapsed_s": total_elapsed,
        "per_task": per_task,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    default=["B7d_AnnotatedNoFraming_NoInt"])
    ap.add_argument("--task_indices", type=int, nargs="*",
                    default=list(range(21)))
    ap.add_argument("--log_dir",
                    default="/home/user/fedcausalworld/experiments/exp1_run/tau_bench_r2_b7d")
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    all_results = {}
    cumulative_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        print(f"\n=== {baseline_id} on τ-bench retail (n={len(args.task_indices)}) ===", flush=True)
        r = run_one_baseline(baseline_id, args.task_indices, args.log_dir)
        all_results[baseline_id] = r
        cumulative_usd += r["total_usd"]
        print(f"  → TS={r['task_success_rate_pp']}pp, cost=${r['total_usd']:.4f}", flush=True)

    elapsed = time.time() - t_global
    summary = {
        "phase": "RUNNING_exp1_r2_b7d_ablation",
        "benchmark": "tau_bench_retail",
        "task_indices": args.task_indices,
        "n_tasks": len(args.task_indices),
        "seeds": 1, "model": MODEL_NAME,
        "elapsed_seconds": elapsed,
        "cumulative_usd": cumulative_usd,
        "per_baseline": all_results,
    }
    if "B7d_AnnotatedNoFraming_NoInt" in all_results:
        b7d = all_results["B7d_AnnotatedNoFraming_NoInt"]["task_success_rate_pp"]
        summary["R2_gate"] = {
            "B7d_TS_pp": b7d,
            "B7_main_TS_pp": 23.81,
            "B2_main_TS_pp": 33.33,
            "B8d_A4_TS_pp": 42.86,
            "B10_main_TS_pp": 52.38,
            "B7d_minus_B7_main_pp": round(b7d - 23.81, 2),
            "B7d_minus_B2_main_pp": round(b7d - 33.33, 2),
            "B7d_minus_B8d_pp": round(b7d - 42.86, 2),
            "recovers_at_least_B2": b7d >= 33.33,
            "G1_5pp_pass": (b7d - 33.33) >= 5.0,
        }
    out_path = os.path.join(args.log_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(json.dumps({k:v for k,v in summary.items() if k!='per_baseline'}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
