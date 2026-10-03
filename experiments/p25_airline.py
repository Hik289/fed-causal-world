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
        kwargs["temperature"] = 0.0
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


ANNOTATED_EDGES_AIRLINE = (
    "account→reservation (authentication gates create_reservation), "
    "reservation→payment (reservation_placed triggers authorize_payment), "
    "reservation→seat_inventory (flight selected triggers seat_hold), "
    "seat_inventory→reservation (seat_held drives reservation_status pending→confirmed), "
    "payment→reservation (payment_captured drives reservation_status confirmed→booked), "
    "reservation→checkin (booked enables checkin), "
    "checkin→boarding_pass (checkin generates boarding_pass), "
    "boarding_pass→reservation (boarded drives reservation_status booked→completed), "
    "cancellation→reservation (cancel_within_fare_rule allows cancel), "
    "cancellation→payment (cancel_eligibility allows refund), "
    "payment→reservation (refund_issued drives reservation_status booked→cancelled), "
    "reservation→seat_inventory (cancelled releases seat_held), "
    "account→checkin (frequent_flyer_tier modulates checkin priority)."
)


BASELINE_HEADERS = {
    "B2_GlobalSeqWM": (
        "WORLD MODEL: GLOBAL SEQUENCE. Track global event sequence across "
        "modules (account, reservation, payment, seat_inventory, checkin, "
        "cancellation)."
    ),
    "B7d_AnnotatedNoFraming_NoInt": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the airline system "
        "(account, reservation, payment, seat_inventory, checkin, "
        "cancellation), inferred from observational traces:\n"
        + ANNOTATED_EDGES_AIRLINE +
        "\nUse the dependency list to anticipate downstream effects."
    ),
    "B8d_AnnotatedNoFraming": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the airline system "
        "(account, reservation, payment, seat_inventory, checkin, "
        "cancellation):\n"
        + ANNOTATED_EDGES_AIRLINE +
        "\nUse the dependency list to anticipate downstream module effects."
    ),
    "B9_AnnotatedNoFramingNoControl": (
        "OPERATIONAL DEPENDENCIES across the 6 modules of the airline system "
        "(account, reservation, payment, seat_inventory, checkin, "
        "cancellation):\n"
        + ANNOTATED_EDGES_AIRLINE +
        "\nThese dependencies describe how the modules interact."
    ),
    "B10_OracleCausalWM": (
        "WORLD MODEL: ORACLE causal graph (ground-truth airline schema). "
        + ANNOTATED_EDGES_AIRLINE +
        " Use this oracle graph for causal rollout."
    ),
}


from tau_bench.envs.airline.env import MockAirlineDomainEnv
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.user import UserStrategy


class PromptHeaderAgent(ToolCallingAgent):
    def __init__(self, header: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wiki = f"{header}\n\n---\n\n{self.wiki}"


def run_baseline(baseline_id, task_indices, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    pred_path = os.path.join(log_dir, f"{baseline_id}_predictions.jsonl")
    fp = open(pred_path, "w")
    header = BASELINE_HEADERS[baseline_id]
    per_task = []
    t_start = time.time()
    for idx in task_indices:
        t0 = time.time()
        _usage_log.clear()
        try:
            env = MockAirlineDomainEnv(user_strategy=UserStrategy.LLM,
                                        user_model=MODEL_NAME,
                                        user_provider="openai",
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
        rec = {"task_idx": idx, "baseline": baseline_id, "reward": reward,
               "task_success": bool(reward >= 0.5),
               "n_llm_calls": len(_usage_log), "task_usd": usd_sum,
               "elapsed_s": elapsed, "error": err}
        fp.write(json.dumps(rec) + "\n"); fp.flush()
        per_task.append(rec)
        print(f"  [{baseline_id}] task={idx} rwd={reward:.2f} "
              f"usd={usd_sum:.4f} t={elapsed:.1f}s", flush=True)
    fp.close()
    n = len(per_task); successes = sum(1 for r in per_task if r["task_success"])
    return {"baseline": baseline_id, "n_tasks": n,
            "task_success_count": successes,
            "task_success_rate_pp": round(100.0 * successes / max(1, n), 2),
            "total_usd": sum(r["task_usd"] for r in per_task),
            "elapsed_s": time.time() - t_start,
            "per_task": per_task}


def wilson_ci_halfwidth(p_hat, n, z=1.96):
    if n == 0: return 0.0
    p = p_hat / 100.0
    denom = 1 + z**2 / n
    margin = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return margin * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    default=["B2_GlobalSeqWM", "B7d_AnnotatedNoFraming_NoInt",
                             "B8d_AnnotatedNoFraming",
                             "B9_AnnotatedNoFramingNoControl",
                             "B10_OracleCausalWM"])
    ap.add_argument("--task_indices", type=int, nargs="+",
                    default=list(range(21)))
    ap.add_argument("--log_dir",
                    default="runs/p25_airline")
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    all_results = {}
    cum_usd = 0.0
    t_global = time.time()
    for b in args.baselines:
        print(f"\n=== {b} on τ-bench airline ===", flush=True)
        r = run_baseline(b, args.task_indices, args.log_dir)
        all_results[b] = r
        cum_usd += r["total_usd"]
        print(f"  → TS={r['task_success_rate_pp']}, cum=${cum_usd:.4f}", flush=True)

    print("\n=== AGGREGATE ===")
    print(f"{'Baseline':<35} {'TS':>8} {'CI':>8}")
    aggregate = {}
    for b, r in all_results.items():
        mean = r["task_success_rate_pp"]
        ci = wilson_ci_halfwidth(mean, r["n_tasks"])
        aggregate[b] = {"TS_pp": mean, "wilson_CI_halfwidth_pp": round(ci, 2)}
        print(f"{b:<35} {mean:>8.2f} {ci:>+8.2f}")

    summary = {"phase": "P2.5_airline_cross_val",
               "baselines": args.baselines,
               "task_indices": args.task_indices,
               "elapsed_seconds": time.time() - t_global,
               "cumulative_usd": cum_usd,
               "per_baseline": all_results,
               "aggregate": aggregate}
    out = os.path.join(args.log_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}, cost ${cum_usd:.4f}")


if __name__ == "__main__":
    sys.exit(main())
