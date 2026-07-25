"""
P1.3: B8e = B8d + "These 14 edges were validated by ICP F-test on N=120
intervention-response matched events with P_verify ≥ 0.95" framing.

Goal: test if making the ICP-validation provenance EXPLICIT in the prompt
moves B8d performance. If B8e > B8d significantly, the "intervention
validation" component is contributing. If equal, anchor_4 finding stands
(annotation alone matters; framing word for validation source doesn't move
the agent).

Same harness as P1.1 (τ-bench retail × 21 × 3 seeds).
"""

import argparse, json, os, sys, time, traceback
from typing import Any, Dict, List
import numpy as np

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


def _patched_completion(*args, **kwargs):
    if AZURE_API_KEY:
        kwargs.setdefault("api_key", AZURE_API_KEY)
    if AZURE_API_BASE:
        kwargs.setdefault("api_base", AZURE_API_BASE)
    if not kwargs.get("model", "").endswith(DEPLOYMENT):
        kwargs["model"] = MODEL_NAME
    kwargs.setdefault("custom_llm_provider", "openai")
    msgs = kwargs.get("messages", [])
    sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
    if "user interacting" in sys_msg.lower() or "simulate the user" in sys_msg.lower():
        kwargs["temperature"] = 0.7
        kwargs["seed"] = _current_seed
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


BASELINE_HEADERS = {
    "B8e_ICPValidated": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the retail system "
        "(account, order, payment, inventory, shipment, refund). The "
        "following 14 edges were statistically validated by ICP "
        "(Invariant Causal Prediction) F-test on N=120 federated "
        "intervention-response matched events with P_verify ≥ 0.95:\n"
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
        super().__init__(*args, **kwargs)
        self.wiki = f"{header}\n\n---\n\n{self.wiki}"


def run_baseline_seed(baseline_id, seed, task_indices, log_dir):
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
            env = MockRetailDomainEnv(user_strategy=UserStrategy.LLM,
                                       user_model=MODEL_NAME, user_provider="openai",
                                       task_split="test", task_index=idx)
            agent = PromptHeaderAgent(header=header, tools_info=env.tools_info,
                                       wiki=env.wiki, model=MODEL_NAME,
                                       provider="openai", temperature=0.0)
            result = agent.solve(env, task_index=idx, max_num_steps=30)
            reward = float(result.reward)
            err = None
        except Exception as exc:
            reward = 0.0
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        elapsed = time.time() - t0
        usd_sum = sum(c["usd"] for c in _usage_log)
        rec = {"task_idx": idx, "baseline": baseline_id, "seed": seed,
               "reward": reward, "task_success": bool(reward >= 0.5),
               "n_llm_calls": len(_usage_log), "task_usd": usd_sum,
               "elapsed_s": elapsed, "error": err}
        fp.write(json.dumps(rec) + "\n"); fp.flush()
        per_task.append(rec)
        print(f"  [{baseline_id} seed={seed}] task={idx} rwd={reward:.2f} "
              f"calls={rec['n_llm_calls']} usd={usd_sum:.4f}", flush=True)
    fp.close()
    n = len(per_task); successes = sum(1 for r in per_task if r["task_success"])
    return {"baseline": baseline_id, "seed": seed, "n_tasks": n,
            "task_success_count": successes,
            "task_success_rate_pp": round(100.0 * successes / max(1, n), 2),
            "total_usd": sum(r["task_usd"] for r in per_task),
            "total_elapsed_s": time.time() - t_start,
            "per_task": per_task}


def wilson_ci_halfwidth(p_hat, n, z=1.96):
    if n == 0: return 0.0
    p = p_hat / 100.0
    denom = 1 + z**2 / n
    margin = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return margin * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--task_indices", type=int, nargs="+",
                    default=list(range(21)))
    ap.add_argument("--log_dir",
                    default="runs/p13_b8e_icp")
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    all_results = []
    cum_usd = 0.0
    t_global = time.time()
    for seed in args.seeds:
        print(f"\n=== B8e_ICPValidated seed={seed} ===", flush=True)
        r = run_baseline_seed("B8e_ICPValidated", seed, args.task_indices, args.log_dir)
        all_results.append(r)
        cum_usd += r["total_usd"]
        print(f"  → seed {seed}: TS={r['task_success_rate_pp']}, "
              f"cum=${cum_usd:.4f}", flush=True)

    ts_seeds = [r["task_success_rate_pp"] for r in all_results]
    total_n = sum(r["n_tasks"] for r in all_results)
    total_won = sum(r["task_success_count"] for r in all_results)
    mean = 100.0 * total_won / max(1, total_n)
    ci = wilson_ci_halfwidth(mean, total_n)
    print(f"\nB8e: per-seed TS={ts_seeds}, pooled mean={mean:.2f}pp ± {ci:.2f}pp")

    summary = {"phase": "P1.3_B8e_ICP_validated",
               "seeds": args.seeds,
               "elapsed_seconds": time.time() - t_global,
               "cumulative_usd": cum_usd,
               "per_seed": all_results,
               "aggregate": {"ts_per_seed": ts_seeds, "mean_TS_pooled": round(mean, 2),
                             "wilson_CI_halfwidth_pp": round(ci, 2),
                             "total_n": total_n, "total_won": total_won}}
    out = os.path.join(args.log_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}, cost ${cum_usd:.4f}")


if __name__ == "__main__":
    sys.exit(main())
