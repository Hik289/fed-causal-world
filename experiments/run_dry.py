"""
run_dry.py — 5+5+2 task dry-run cost calibration.

Per experiment design dispatch:
  - τ-bench retail × 5 tasks
  - ALFWorld    × 5 tasks
  - AndroidWorld× 2 tasks
  - Baseline: B2 (Global Sequence WM, cheapest)
  - Goal: measure real Azure GPT-5.4-mini cost/task → extrapolate to full
    8-experiment budget, compare to EDA A4 $535 estimate.

Outputs:
  - dry_run_out/predictions.jsonl    (one line per task)
  - dry_run_out/summary.json         (per-benchmark + global stats)

If total dry-run cost extrapolates above $500 budget cap, exit 2 (the runner will
escalate to User).
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import List, Dict, Any

from event_traces import synth_task_pool, BENCHMARKS
from baselines.b2_global_seq_wm import B2_GlobalSeqWM
from llm_client import get_counter, reset_counter
from metrics import aggregate


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dry_run_out")
os.makedirs(OUT_DIR, exist_ok=True)


DRY_RUN_PLAN = [
    ("tau_bench",    5),
    ("alfworld",     5),
    ("androidworld", 2),
]

# Full experiment volume estimate (matches EDA §8 #7 + hypothesis.md
# cost_estimate summing).
FULL_EXP_TASK_COUNT = {
    # benchmark -> (n_tasks per main eval, total baseline-seed-runs over Exp1-8)
    # See exp_design.md §6 for derivation.
    "tau_bench":    21 * 3 * 12,    # 21 test tasks × 3 seeds × 12 baselines
    "alfworld":     268 * 3 * 12,   # ALFWorld unseen
    "androidworld": 12 * 3 * 12,    # 12 test tasks × 3 seeds × 12 baselines
}

BUDGET_FLOOR_USD = 300.0
BUDGET_CAP_USD = 500.0


def run_dry() -> Dict[str, Any]:
    reset_counter()
    baseline = B2_GlobalSeqWM()
    predictions_path = os.path.join(OUT_DIR, "predictions.jsonl")

    bench_records: Dict[str, List[Dict[str, Any]]] = {b: [] for b, _ in DRY_RUN_PLAN}
    bench_usage: Dict[str, Dict[str, float]] = {b: {"calls": 0, "prompt_tokens": 0,
                                                     "completion_tokens": 0, "usd": 0.0,
                                                     "elapsed_s": 0.0}
                                                 for b, _ in DRY_RUN_PLAN}

    t_start = time.time()
    with open(predictions_path, "w") as fp:
        for benchmark, n_tasks in DRY_RUN_PLAN:
            print(f"=== {benchmark}: {n_tasks} tasks (B2_GlobalSeqWM) ===")
            tasks = synth_task_pool(benchmark, n_tasks, seed=0)
            for task in tasks:
                t0 = time.time()
                try:
                    pred, usage = baseline.predict_next_event(task, max_output_tokens=64)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERR {task.task_id}: {exc}")
                    pred = {"module_id": None, "event_type": None, "error": str(exc)}
                    usage = {"prompt_tokens": 0, "completion_tokens": 0,
                             "usd_estimated": 0.0, "elapsed_s": time.time() - t0}
                target = {"module_id": task.target_event.module_id,
                          "event_type": task.target_event.event_type}
                rec = {
                    "task_id": task.task_id,
                    "benchmark": benchmark,
                    "baseline": baseline.name,
                    "pred": pred,
                    "target": target,
                    "usage": usage,
                }
                fp.write(json.dumps(rec) + "\n")
                bench_records[benchmark].append(rec)
                bench_usage[benchmark]["calls"] += 1
                bench_usage[benchmark]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                bench_usage[benchmark]["completion_tokens"] += usage.get("completion_tokens", 0)
                bench_usage[benchmark]["usd"] += usage.get("usd_estimated", 0.0)
                bench_usage[benchmark]["elapsed_s"] += usage.get("elapsed_s", 0.0)
                print(f"  {task.task_id}: pred={pred.get('module_id')}.{pred.get('event_type')}"
                      f"  target={target['module_id']}.{target['event_type']}"
                      f"  usd={usage.get('usd_estimated', 0):.6f}"
                      f"  pt={usage.get('prompt_tokens')}, ct={usage.get('completion_tokens')}")

    total_elapsed = time.time() - t_start
    counter = get_counter()

    # Per-benchmark aggregate + extrapolation
    summary = {
        "dry_run_plan": DRY_RUN_PLAN,
        "total_elapsed_s": total_elapsed,
        "counter": counter,
        "per_benchmark": {},
    }
    extrapolated_total_usd = 0.0
    for benchmark, n_tasks in DRY_RUN_PLAN:
        recs = bench_records[benchmark]
        usage = bench_usage[benchmark]
        agg = aggregate(recs)
        avg_usd = usage["usd"] / max(1, usage["calls"])
        n_full = FULL_EXP_TASK_COUNT[benchmark]
        proj_b2 = avg_usd * n_full
        proj_mix = proj_b2 * 1.5
        extrapolated_total_usd += proj_mix
        summary["per_benchmark"][benchmark] = {
            "n_tasks": n_tasks,
            "avg_usd_per_task": avg_usd,
            "avg_prompt_tokens": usage["prompt_tokens"] / max(1, usage["calls"]),
            "avg_completion_tokens": usage["completion_tokens"] / max(1, usage["calls"]),
            "avg_elapsed_s": usage["elapsed_s"] / max(1, usage["calls"]),
            "transition_em": agg.get("transition_em"),
            "parse_success_rate": agg.get("parse_success_rate"),
            "projected_full_exp_tasks": n_full,
            "projected_b2_only_usd": proj_b2,
            "projected_b0_to_b11_usd": proj_mix,
        }

    summary["extrapolated_total_usd"] = extrapolated_total_usd
    summary["eda_a4_estimate_usd"] = 535.0
    summary["budget_floor_usd"] = BUDGET_FLOOR_USD
    summary["budget_cap_usd"] = BUDGET_CAP_USD
    summary["over_budget"] = bool(extrapolated_total_usd > BUDGET_CAP_USD)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== DRY RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run_dry()
    if s["over_budget"]:
        print(f"!!! OVER BUDGET: projected ${s['extrapolated_total_usd']:.2f} > "
              f"${BUDGET_CAP_USD:.2f}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)
