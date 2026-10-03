from __future__ import annotations
import json
import os
import time
from typing import Dict, Any

from event_traces import synth_task_pool
from baselines.b2_global_seq_wm import B2_GlobalSeqWM
from llm_client import chat, reset_counter, get_counter


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dry_run_out")

MULTITURN_PLAN = [
    ("tau_bench",    2, 15),
    ("alfworld",     2, 30),
    ("androidworld", 1, 25),
]

FULL_EXP_TASK_COUNT = {
    "tau_bench":    21 * 3 * 12,
    "alfworld":     268 * 3 * 12,
    "androidworld": 12 * 3 * 12,
}

BUDGET_CAP_USD = 500.0


def run_multiturn() -> Dict[str, Any]:
    reset_counter()
    baseline = B2_GlobalSeqWM()
    summary = {"plan": MULTITURN_PLAN, "per_benchmark": {}}
    log_path = os.path.join(OUT_DIR, "multiturn_predictions.jsonl")
    t_start = time.time()
    extrapolated = 0.0
    with open(log_path, "w") as fp:
        for benchmark, n_tasks, n_calls in MULTITURN_PLAN:
            print(f"\n=== {benchmark}: {n_tasks} tasks × {n_calls} calls/task ===")
            tasks = synth_task_pool(benchmark, n_tasks, seed=1)
            usd_per_task = []
            pt_per_task = []
            ct_per_task = []
            time_per_task = []
            for task in tasks:
                t0 = time.time()
                task_usd = 0.0
                task_pt = 0
                task_ct = 0
                for call_i in range(n_calls):
                    prefix_len = min(len(task.events), max(2, call_i + 2))
                    sub_task = task
                    sub_task_events = list(task.events[:prefix_len])
                    from baselines.base import render_event_history, render_module_list
                    from event_traces import Task as _Task
                    sub = _Task(task_id=task.task_id, benchmark=task.benchmark,
                                events=sub_task_events, target_event=task.target_event,
                                ground_truth_edges=task.ground_truth_edges)
                    prompt = baseline.build_prompt(sub)
                    messages = [
                        {"role": "system",
                         "content": "You are a world model. Predict the next event in strict JSON. "
                                    "Output only {\"module_id\": \"...\", \"event_type\": \"...\"}."},
                        {"role": "user", "content": prompt},
                    ]
                    try:
                        text, usage = chat(messages, max_tokens=64, temperature=0.0)
                    except Exception as exc:
                        print(f"  ERR call {call_i}: {exc}")
                        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                                 "usd_estimated": 0.0}
                    task_usd += usage.get("usd_estimated", 0.0)
                    task_pt += usage.get("prompt_tokens", 0)
                    task_ct += usage.get("completion_tokens", 0)
                elapsed = time.time() - t0
                fp.write(json.dumps({
                    "task_id": task.task_id, "benchmark": benchmark, "n_calls": n_calls,
                    "task_usd": task_usd, "task_prompt_tokens": task_pt,
                    "task_completion_tokens": task_ct, "elapsed_s": elapsed,
                }) + "\n")
                usd_per_task.append(task_usd)
                pt_per_task.append(task_pt)
                ct_per_task.append(task_ct)
                time_per_task.append(elapsed)
                print(f"  {task.task_id}: usd={task_usd:.5f}, pt={task_pt}, "
                      f"ct={task_ct}, t={elapsed:.1f}s")

            avg_usd = sum(usd_per_task) / len(usd_per_task)
            n_full = FULL_EXP_TASK_COUNT[benchmark]
            proj_b2 = avg_usd * n_full
            proj_mix = proj_b2 * 1.5
            extrapolated += proj_mix
            summary["per_benchmark"][benchmark] = {
                "n_tasks": n_tasks, "n_calls_per_task": n_calls,
                "avg_usd_per_task": avg_usd,
                "avg_prompt_tokens_per_task": sum(pt_per_task) / len(pt_per_task),
                "avg_completion_tokens_per_task": sum(ct_per_task) / len(ct_per_task),
                "avg_elapsed_s_per_task": sum(time_per_task) / len(time_per_task),
                "projected_full_exp_tasks_b2_equiv": n_full,
                "projected_b2_only_usd": proj_b2,
                "projected_b0_to_b11_usd": proj_mix,
            }

    summary["total_elapsed_s"] = time.time() - t_start
    summary["counter"] = get_counter()
    summary["extrapolated_total_usd"] = extrapolated
    summary["eda_a4_estimate_usd"] = 535.0
    summary["budget_cap_usd"] = BUDGET_CAP_USD
    summary["over_budget"] = bool(extrapolated > BUDGET_CAP_USD)

    with open(os.path.join(OUT_DIR, "multiturn_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MULTI-TURN DRY RUN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_multiturn()
