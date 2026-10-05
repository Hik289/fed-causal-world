from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dry_run_out")


DRY_RUN_PLAN = [
    ("tau_bench",    5),
    ("alfworld",     5),
    ("androidworld", 2),
]

FULL_EXP_TASK_COUNT = {
    "tau_bench":    21 * 3 * 12,
    "alfworld":     268 * 3 * 12,
    "androidworld": 12 * 3 * 12,
}

BUDGET_FLOOR_USD = 300.0
BUDGET_CAP_USD = 500.0


def run_dry() -> Dict[str, Any]:
    source_directory = Path(__file__).resolve().parents[1] / "src" / "fed_causal"
    sys.path.insert(0, str(source_directory))
    from event_traces import synth_task_pool
    from baselines.b2_global_seq_wm import B2_GlobalSeqWM
    from llm_client import get_counter, reset_counter
    from metrics import aggregate

    os.makedirs(OUT_DIR, exist_ok=True)
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
                except Exception as exc:
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


EXPERIMENTS = {
    "theory": ("experiments.p15_synthetic_3seed", "theory_sweeps_main", "Confounding, proxies, depth, coverage, and spectral stability."),
    "interfaces": ("fed_causal.pipeline", "construct_interfaces_main", "Interface recovery and construction cost from supplied traces."),
    "modular": ("experiments.exp1_taubench", "modular_evaluation_main", "ALFWorld and tau-bench graph, control, attention, and feedback evaluations."),
    "scienceworld": ("experiments.exp1_taubench", "scienceworld_evaluation_main", "ScienceWorld task scores with supplied interface graphs."),
    "apibank": ("experiments.exp1_taubench", "apibank_evaluation_main", "APIBank persistent-state dialogue diagnostics."),
    "summarize": ("fed_causal.metrics", "summarize_agent_runs_main", "Aggregate saved outputs and compare matched tasks."),
}


def paper_experiments_main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Experiment entrypoints in the original repository files. External data and configurations are required; published results are not embedded.",
        epilog="Pass an experiment name followed by --help for its arguments. With no arguments, run_dry.py retains its original pilot experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiment", choices=EXPERIMENTS,
                        help="\n".join(f"{key}: {value[2]}" for key, value in EXPERIMENTS.items()))
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return
    selected = parser.parse_args(arguments[:1])
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "src"))
    module, entrypoint, _ = EXPERIMENTS[selected.experiment]
    getattr(importlib.import_module(module), entrypoint)(arguments[1:])


if __name__ == "__main__" and len(sys.argv) > 1:
    paper_experiments_main()
elif __name__ == "__main__":
    s = run_dry()
    if s["over_budget"]:
        print(f"!!! OVER BUDGET: projected ${s['extrapolated_total_usd']:.2f} > "
              f"${BUDGET_CAP_USD:.2f}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)
