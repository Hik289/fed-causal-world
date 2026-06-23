"""
anchor1_b2_repro.py — H0.anchor_1: B2 (Global Sequence WM = LLM-as-WM via
tool-calling agent) reproduction on vanilla τ-bench retail.

user gate: Task Success ∈ [35%, 60%] (already-published GPT-4o = 40-55%,
± 5pp tolerance for GPT-5.4-mini vs GPT-4o; specification §3).

Implementation:
  - Use τ-bench's ToolCallingAgent (acts as B2: full global tool-calling
    sequence model) on retail test split.
  - Monkey-patch litellm.completion to inject Azure key + base inline
    (API config: NO env vars, credentials only inline in code).
  - User simulator and agent both use gpt-5.4-mini at temp=0.
  - 5 task subset for cost; optional B0 (ReAct/chat agent) parallel for harness
    sanity check.

Output:
  - your-server: anchor_1_run/predictions.jsonl  (per-task reward + cost + steps)
  - GCP:    results/anchor_1_tau_bench_b2_repro.json (aggregate)
"""

from __future__ import annotations
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

# === API config: INLINE CREDS (NO env vars, file is .gitignored) ===
AZURE_API_KEY = "YOUR_AZURE_API_KEY"
AZURE_API_BASE = "YOUR_AZURE_ENDPOINT"
MODEL_NAME = "openai/gpt-5.4-mini"   # litellm openai-compatible provider routing
DEPLOYMENT = "gpt-5.4-mini"

PRICE_INPUT_PER_1M = 0.25     # USD / 1M input tokens (gpt-5.4-mini)
PRICE_OUTPUT_PER_1M = 2.00    # USD / 1M output tokens


# ---------------------------------------------------------------------------
# Monkey-patch litellm.completion: inject api_key + api_base on every call so
# τ-bench's ToolCallingAgent and LLMUserSimulationEnv route to Azure.
# ---------------------------------------------------------------------------

import litellm

_original_completion = litellm.completion

# Per-call usage tracker (litellm doesn't always populate response_cost on Azure)
_usage_log: List[Dict[str, Any]] = []


def _patched_completion(*args, **kwargs):
    # Force our Azure endpoint
    kwargs["api_key"] = AZURE_API_KEY
    kwargs["api_base"] = AZURE_API_BASE
    # Force the same model (tau-bench passes model="openai/gpt-5.4-mini" already
    # via our --model flag).  If a different model slips in (e.g. user-model),
    # still route to our deployment.
    if not kwargs.get("model", "").endswith(DEPLOYMENT):
        kwargs["model"] = MODEL_NAME
    # litellm.completion drops custom_llm_provider=None on some paths; ensure
    # openai-compatible routing.
    kwargs.setdefault("custom_llm_provider", "openai")
    t0 = time.time()
    res = _original_completion(*args, **kwargs)
    elapsed = time.time() - t0
    # Extract usage + cost
    try:
        u = res.usage
        pt = getattr(u, "prompt_tokens", 0) or 0
        ct = getattr(u, "completion_tokens", 0) or 0
        usd = (pt * PRICE_INPUT_PER_1M + ct * PRICE_OUTPUT_PER_1M) / 1_000_000.0
        # Force response_cost so tau_bench accounting works
        if hasattr(res, "_hidden_params"):
            res._hidden_params["response_cost"] = usd
        else:
            res._hidden_params = {"response_cost": usd}
        _usage_log.append({
            "prompt_tokens": pt, "completion_tokens": ct,
            "usd": usd, "elapsed_s": elapsed,
        })
    except Exception:  # noqa: BLE001
        pass
    return res


litellm.completion = _patched_completion


# ---------------------------------------------------------------------------
# Run a small subset of vanilla τ-bench retail tasks
# ---------------------------------------------------------------------------

from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.user import UserStrategy


def run_anchor_1(task_indices: List[int], log_dir: str) -> Dict[str, Any]:
    os.makedirs(log_dir, exist_ok=True)
    pred_path = os.path.join(log_dir, "predictions.jsonl")
    fp = open(pred_path, "w")

    per_task = []
    t_global = time.time()
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
            agent = ToolCallingAgent(
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
        except Exception as exc:  # noqa: BLE001
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
        print(f"task_idx={idx:>3} reward={reward:.2f} ts={rec['task_success']} "
              f"n_calls={rec['n_llm_calls']:>3} usd={usd_sum:.4f} "
              f"elapsed={elapsed:.1f}s err={err or '-'}")

    fp.close()
    total_elapsed = time.time() - t_global

    # Aggregate
    n = len(per_task)
    successes = sum(1 for r in per_task if r["task_success"])
    avg_usd = sum(r["task_usd"] for r in per_task) / max(1, n)
    parse_fail = sum(1 for r in per_task if r["error"] is not None)
    summary = {
        "phase": "RUNNING_anchor_1",
        "subphase": "anchor_1_tau_bench_b2_repro",
        "model": MODEL_NAME,
        "deployment": DEPLOYMENT,
        "n_tasks": n,
        "task_indices": task_indices,
        "task_success_count": successes,
        "task_success_rate_pp": round(100.0 * successes / max(1, n), 2),
        "harness_failure_count": parse_fail,
        "avg_usd_per_task": avg_usd,
        "total_usd": sum(r["task_usd"] for r in per_task),
        "avg_llm_calls_per_task": (
            sum(r["n_llm_calls"] for r in per_task) / max(1, n)),
        "avg_elapsed_s_per_task": (
            sum(r["elapsed_s"] for r in per_task) / max(1, n)),
        "total_elapsed_s": total_elapsed,
        "per_task": per_task,
        "gate_lower_pp": 35.0,
        "gate_upper_pp": 60.0,
    }
    summary["gate_pass"] = (summary["gate_lower_pp"]
                            <= summary["task_success_rate_pp"]
                            <= summary["gate_upper_pp"])

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {summary_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"},
                     indent=2))
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_indices", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--log_dir", default="/home/user/fedcausalworld/experiments/anchor_1_run")
    args = ap.parse_args()
    s = run_anchor_1(args.task_indices, args.log_dir)
    sys.exit(0 if s["gate_pass"] else 2)


if __name__ == "__main__":
    main()
