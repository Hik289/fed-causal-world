from __future__ import annotations
import json
import os
import sys
import time
import traceback
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



ANNOTATED_EDGES_RETAIL = (
    "Cross-module causal edges (each with operational meaning): "
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


BASELINE_HEADERS_A3 = {
    "B8c_AnnotatedNoConstraint": (
        "WORLD MODEL: FedCausalCompose — causal graph composed from local "
        "modular mechanisms. The retail system has 6 functional modules: "
        "account, order, payment, inventory, shipment, refund. "
        + ANNOTATED_EDGES_RETAIL +
        " Use this causal graph to PREDICT downstream module effects when "
        "executing each tool call. The graph was validated by federated "
        "intervention-response matching."
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
    header = BASELINE_HEADERS_A3[baseline_id]
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
        calls_used = list(_usage_log)
        usd_sum = sum(c["usd"] for c in calls_used)
        rec = {
            "task_idx": idx, "baseline": baseline_id,
            "reward": reward, "task_success": bool(reward >= 0.5),
            "n_messages": n_msgs, "n_llm_calls": len(calls_used),
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
                    default=["B8c_AnnotatedNoConstraint"])
    ap.add_argument("--task_indices", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--log_dir",
                    default="runs/tau_bench_a3")
    ap.add_argument("--cell_cost_cap", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    all_results = {}
    cumulative_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        print(f"\n=== {baseline_id} on τ-bench retail (n={len(args.task_indices)}) ===", flush=True)
        try:
            r = run_one_baseline(baseline_id, args.task_indices, args.log_dir)
        except Exception as exc:
            print(f"!!! HARNESS FAILURE: {exc}", flush=True)
            traceback.print_exc()
            continue
        all_results[baseline_id] = r
        cumulative_usd += r["total_usd"]
        print(f"  → TS={r['task_success_rate_pp']}pp, cost=${r['total_usd']:.4f}", flush=True)

    elapsed = time.time() - t_global
    summary = {
        "phase": "RUNNING_exp1_a3_ablation",
        "benchmark": "tau_bench_retail",
        "task_indices": args.task_indices,
        "n_tasks": len(args.task_indices),
        "seeds": 1, "model": MODEL_NAME,
        "elapsed_seconds": elapsed,
        "cumulative_usd": cumulative_usd,
        "per_baseline": all_results,
    }
    if "B8c_AnnotatedNoConstraint" in all_results:
        b8c = all_results["B8c_AnnotatedNoConstraint"]["task_success_rate_pp"]
        summary["A3_gate"] = {
            "B8c_TS_pp": b8c,
            "B8c_minus_B2_main_pp": round(b8c - 33.33, 2),
            "B8c_minus_B10_main_pp": round(b8c - 52.38, 2),
            "G1_5pp_pass": (b8c - 33.33) >= 5.0,
        }
    out_path = os.path.join(args.log_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(json.dumps({k:v for k,v in summary.items() if k!='per_baseline'}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
