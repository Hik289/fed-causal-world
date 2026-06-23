"""
exp1_taubench.py — Exp1 on vanilla τ-bench retail, 9 baselines × 21 tasks × 1 seed.

Implementation strategy: PROMPT-MODULATION OF B2 anchor_1 HARNESS.

  anchor_1 worked: ToolCallingAgent + LLM user simulator on τ-bench retail.
  All baselines (B0/B1/B2/B3/B4/B5/B7/B8/B10) share the same tool-calling
  scaffolding, differing ONLY in the system-prompt header prepended to the
  env.wiki.  This keeps the implementation honest:
   - All baselines see the same tools, same user, same reward
   - The only thing that changes is the WORLD-MODEL framing in the prompt
   - This is exactly what exp_design.md §1 specified

Baseline prompt-headers (per exp_design.md §1):
  B0 No WM         : "Do not build a world model. Act greedily."
  B1 Local WM      : "Maintain a LOCAL world model per module only."
  B2 Global Seq WM : "Maintain a GLOBAL SEQUENCE world model over events."
  B3 Transition Graph
  B4 Correlation WM
  B5 Temporal WM
  B7 Causal-no-int : "Causal graph from observational data, no interventions"
  B8 FedCausalCompose: full causal + decentralized control
  B10 Oracle Causal: ground-truth causal graph + control

Note: specification listed B0/B1/B2/B3/B4/B5/B7/B10 (8 baseline). PASS criterion
"B8 vs B2 ≥ +5pp" requires B8 to also be present. I add B8 → 9 baselines total
(flagged in completion report for user awareness).

Modular awareness: per user, modularization_spec.md 6-module decomposition
is fed to B8 only (the only baseline that explicitly reasons about modules).
"""

from __future__ import annotations
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

# === API config: INLINE CREDS (file is .gitignored) ===
AZURE_API_KEY = "YOUR_AZURE_API_KEY"
AZURE_API_BASE = "YOUR_AZURE_ENDPOINT"
MODEL_NAME = "openai/gpt-5.4-mini"
DEPLOYMENT = "gpt-5.4-mini"

PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00


# ---------------------------------------------------------------------------
# Patch litellm.completion: inject Azure creds + force model on every call
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Baseline-specific prompt headers (prepended to env.wiki for ToolCallingAgent)
# ---------------------------------------------------------------------------

MODULARIZATION_SUMMARY = (
    "The retail system has 6 functional modules: account, order, payment, "
    "inventory, shipment, refund. Cross-module dependencies (validated edges): "
    "account→order, order→payment, order→inventory, payment→order, "
    "inventory→order, order→shipment, payment→shipment, shipment→order, "
    "shipment→refund, refund→payment, payment→order."
)

ORACLE_EDGES = (
    "GROUND-TRUTH causal interface edges (Oracle): "
    "account→order (auth gates create_order), order→payment (order_placed "
    "triggers authorize), order→inventory (order_placed triggers reserve), "
    "inventory→order (inventory_reserved drives pending→draft), payment→order "
    "(payment_captured drives pending→confirmed), order→shipment "
    "(order_confirmed triggers label), inventory+payment→shipment (AND "
    "mediator), shipment→order (shipment_completed drives shipped→delivered), "
    "shipment→refund (delayed refund_eligibility), refund→payment "
    "(refund_issued triggers refund), payment→order (refunded triggers "
    "returned), order→inventory (cancel triggers release), order→payment "
    "(cancel voids auth), account→shipment (address modulates delivery_days)."
)

BASELINE_HEADERS = {
    "B0_NoWM": (
        "WORLD MODEL: None. You do NOT maintain or use any world model. "
        "Act greedily based on the current observation only. Do not predict "
        "future states or downstream effects."
    ),
    "B1_LocalWM": (
        "WORLD MODEL: Per-module LOCAL only. You may track state within "
        "each individual module (e.g. payment status, order status) but do "
        "NOT model cross-module dependencies. Treat each tool call as an "
        "isolated within-module action."
    ),
    "B2_GlobalSeqWM": (
        "WORLD MODEL: GLOBAL SEQUENCE. You track the global event sequence "
        "across all modules (account, order, payment, inventory, shipment, "
        "refund) and predict next states from this sequence. No explicit "
        "causal structure — purely sequential pattern matching."
    ),
    "B3_TransitionGraph": (
        "WORLD MODEL: GLOBAL TRANSITION GRAPH. You maintain a (state, action) "
        "→ next_state transition graph over the full multi-module state "
        "space. Use observed transitions to predict next state. No causal "
        "direction — purely predictive."
    ),
    "B4_CorrelationWM": (
        "WORLD MODEL: CORRELATION-BASED DEPENDENCY. You track which events "
        "co-occur across modules and use these correlations to predict next "
        "events. No causal direction implied — just co-occurrence statistics."
    ),
    "B5_TemporalWM": (
        "WORLD MODEL: TEMPORAL PRECEDENCE. You track temporal ordering and "
        "lag patterns between events across modules (e.g. payment events "
        "typically precede shipment events by 1-3 steps). Use temporal "
        "patterns to predict next steps. No structural causal model."
    ),
    "B7_CausalWMNoInt": (
        "WORLD MODEL: CAUSAL graph from OBSERVATIONAL data only. "
        + MODULARIZATION_SUMMARY +
        " You have a causal interface graph but it was learned from "
        "observation only (NO intervention-based validation). Use the graph "
        "to predict causal effects, but be aware some edges may be confounded "
        "or directionally ambiguous."
    ),
    "B8_FedCausalCompose": (
        "WORLD MODEL: FedCausalCompose — Full causal graph + decentralized "
        "causal control. " + MODULARIZATION_SUMMARY +
        " For each action, INSPECT required upstream interface states, "
        "PREDICT all downstream module effects, BLOCK any action that "
        "violates global causal constraints, and VERIFY both local and "
        "downstream outcomes after execution. The graph is validated by "
        "intervention-response matching with high confidence."
    ),
    "B10_OracleCausalWM": (
        "WORLD MODEL: ORACLE causal graph (ground-truth). " + ORACLE_EDGES +
        " Use this oracle graph to perform causal rollout and downstream-"
        "effect prediction with maximal confidence."
    ),
}


# ---------------------------------------------------------------------------
# Custom agent: wrap ToolCallingAgent with prompt-header prepend
# ---------------------------------------------------------------------------

from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.user import UserStrategy


class PromptHeaderAgent(ToolCallingAgent):
    """Identical to ToolCallingAgent but prepends a baseline-specific
    instruction block to the env.wiki (system prompt)."""

    def __init__(self, header: str, *args, **kwargs):
        # store header BEFORE super().__init__ in case wiki accessed early
        self._baseline_header = header
        super().__init__(*args, **kwargs)
        # rewrite the wiki to include the header
        self.wiki = f"{header}\n\n---\n\n{self.wiki}"


# ---------------------------------------------------------------------------
# Per-baseline runner
# ---------------------------------------------------------------------------

def run_one_baseline(baseline_id: str, task_indices: List[int],
                     log_dir: str) -> Dict[str, Any]:
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
            env = MockRetailDomainEnv(
                user_strategy=UserStrategy.LLM,
                user_model=MODEL_NAME,
                user_provider="openai",
                task_split="test",
                task_index=idx,
            )
            agent = PromptHeaderAgent(
                header=header,
                tools_info=env.tools_info,
                wiki=env.wiki,
                model=MODEL_NAME,
                provider="openai",
                temperature=0.0,
            )
            result = agent.solve(env, task_index=idx, max_num_steps=30)
            reward = float(result.reward)
            n_msgs = len(result.messages)
            total_cost = float(result.total_cost or 0.0)
            err = None
        except Exception as exc:
            reward = 0.0
            n_msgs = 0
            total_cost = 0.0
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        elapsed = time.time() - t0
        calls_used = list(_usage_log)
        usd_sum = sum(c["usd"] for c in calls_used)
        rec = {
            "task_idx": idx,
            "baseline": baseline_id,
            "reward": reward,
            "task_success": bool(reward >= 0.5),
            "n_messages": n_msgs,
            "n_llm_calls": len(calls_used),
            "task_usd": usd_sum,
            "tau_bench_reported_cost": total_cost,
            "elapsed_s": elapsed,
            "error": err,
        }
        fp.write(json.dumps(rec) + "\n"); fp.flush()
        per_task.append(rec)
        print(f"  [{baseline_id}] task={idx:>3} reward={reward:.2f} "
              f"calls={rec['n_llm_calls']:>3} usd={usd_sum:.4f} "
              f"t={elapsed:.1f}s err={err or '-'}")
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
        "avg_calls_per_task": sum(r["n_llm_calls"] for r in per_task) / max(1, n),
        "avg_elapsed_s": sum(r["elapsed_s"] for r in per_task) / max(1, n),
        "total_elapsed_s": total_elapsed,
        "per_task": per_task,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    default=["B0_NoWM", "B1_LocalWM", "B2_GlobalSeqWM",
                             "B3_TransitionGraph", "B4_CorrelationWM",
                             "B5_TemporalWM", "B7_CausalWMNoInt",
                             "B8_FedCausalCompose", "B10_OracleCausalWM"])
    ap.add_argument("--task_indices", type=int, nargs="*",
                    default=list(range(21)))
    ap.add_argument("--log_dir", default="/home/user/fedcausalworld/experiments/exp1_run/tau_bench")
    ap.add_argument("--cell_cost_cap", type=float, default=30.0,
                    help="Per-baseline cost cap; if exceeded, abort and report")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    all_results = {}
    cumulative_usd = 0.0
    t_global = time.time()
    for baseline_id in args.baselines:
        print(f"\n=== {baseline_id} on tau_bench retail (n={len(args.task_indices)} tasks) ===")
        try:
            r = run_one_baseline(baseline_id, args.task_indices, args.log_dir)
        except Exception as exc:
            print(f"!!! HARNESS FAILURE on {baseline_id}: {exc}")
            traceback.print_exc()
            all_results[baseline_id] = {
                "baseline": baseline_id, "error": str(exc),
                "task_success_rate_pp": None,
            }
            continue
        all_results[baseline_id] = r
        cumulative_usd += r["total_usd"]
        print(f"  → TS={r['task_success_rate_pp']}pp, "
              f"cost={r['total_usd']:.4f}, "
              f"cell_avg=${r['avg_usd_per_task']:.4f}/task")
        if r["total_usd"] > args.cell_cost_cap:
            print(f"!!! CELL COST {r['total_usd']:.2f} > CAP "
                  f"{args.cell_cost_cap:.2f} — aborting Exp1 τ-bench")
            break

    elapsed = time.time() - t_global
    summary = {
        "phase": "RUNNING_exp1_w2_d1",
        "benchmark": "tau_bench_retail",
        "task_indices": args.task_indices,
        "n_tasks_per_baseline": len(args.task_indices),
        "seeds": 1,
        "model": MODEL_NAME,
        "elapsed_seconds": elapsed,
        "cumulative_usd": cumulative_usd,
        "per_baseline": all_results,
    }

    # Compute G1 gap (B8 vs B2)
    if "B8_FedCausalCompose" in all_results and "B2_GlobalSeqWM" in all_results:
        b8_ts = all_results["B8_FedCausalCompose"].get("task_success_rate_pp")
        b2_ts = all_results["B2_GlobalSeqWM"].get("task_success_rate_pp")
        if b8_ts is not None and b2_ts is not None:
            summary["G1_proxy_taubench"] = {
                "B8_TS_pp": b8_ts,
                "B2_TS_pp": b2_ts,
                "B8_minus_B2_pp": round(b8_ts - b2_ts, 2),
                "G1_5pp_pass": (b8_ts - b2_ts) >= 5.0,
            }
        if "B10_OracleCausalWM" in all_results:
            b10_ts = all_results["B10_OracleCausalWM"].get("task_success_rate_pp")
            if b10_ts is not None:
                summary["G1_proxy_taubench"]["B10_TS_pp"] = b10_ts
                summary["G1_proxy_taubench"]["B8_vs_B10_gap_pp"] = round(b8_ts - b10_ts, 2)

    out_path = os.path.join(args.log_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(f"Total: ${cumulative_usd:.4f}, elapsed {elapsed:.1f}s")
    if "G1_proxy_taubench" in summary:
        print(f"G1 proxy: {summary['G1_proxy_taubench']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
