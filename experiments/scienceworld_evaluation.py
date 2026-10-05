from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.agent_evaluation_common import add_policy_arguments, create_run_directory, file_sha256
from experiments.agent_evaluation_common import policy_from_args, validate_evaluation_ids, write_json


SOURCE_API = "https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        tasks = json.load(handle)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task manifest must be a nonempty JSON list.")
    identities = set()
    ids = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("task_name"), str):
            raise ValueError("Each task needs task_name and variation_id.")
        variation = task.get("variation_id")
        if isinstance(variation, bool) or not isinstance(variation, int) or variation < 0:
            raise ValueError("variation_id must be a nonnegative integer.")
        identity = (task["task_name"], variation)
        task.setdefault("task_id", f"scienceworld:{task['task_name']}:{variation}")
        if not isinstance(task["task_id"], str) or not task["task_id"] or task["task_id"] in ids or identity in identities:
            raise ValueError("Manifest contains duplicate or invalid task identifiers.")
        identities.add(identity)
        ids.add(task["task_id"])
    return tasks


def available_actions(env: Any, action_space: str) -> list[str]:
    if action_space == "valid":
        actions = env.get_valid_action_object_combinations()
    else:
        templates, _ = env.get_possible_action_object_combinations()
        actions = [item["action"] for item in templates]
    if not actions or not all(isinstance(item, str) and item for item in actions):
        raise ValueError("ScienceWorld returned an empty or malformed action catalogue.")
    return sorted(set(actions))


def validate_tasks(env: Any, tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    names = set(env.get_task_names())
    for task in tasks:
        if task["task_name"] not in names:
            raise ValueError(f"Unknown ScienceWorld task: {task['task_name']}")
        env.load(task["task_name"], 0, args.simplifications, generateGoldPath=False)
        variations = set(getattr(env, f"get_variations_{args.split}")())
        if task["variation_id"] not in variations:
            raise ValueError(f"Task {task['task_id']} is not in the requested {args.split} split.")
        env.load(task["task_name"], task["variation_id"], args.simplifications, generateGoldPath=False)
        observation, info = env.reset()
        if not isinstance(observation, str) or not isinstance(info, dict) or "score" not in info:
            raise ValueError("Unexpected ScienceWorld reset contract.")
        available_actions(env, args.action_space)


def evaluate_task(env: Any, task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    policy = policy_from_args(args)
    result = {"benchmark": "scienceworld", "task_id": task["task_id"], "task_name": task["task_name"],
              "variation_id": task["variation_id"], "seed": args.seed, "condition": args.condition,
              "status": "setup_error", "success": None, "score": None, "reward": None, "steps": 0,
              "trajectory": [], "errors": []}
    try:
        env.load(task["task_name"], task["variation_id"], args.simplifications, generateGoldPath=False)
        observation, info = env.reset()
        goal = env.get_task_description()
        score = float(info["score"])
    except Exception as error:
        result["errors"].append({"phase": "environment_setup", "type": type(error).__name__, "message": str(error)})
        result.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(), model_calls=policy.calls)
        return result
    reward_sum = 0.0
    done = False
    result["status"] = "running"
    phase = "policy"
    try:
        for step in range(args.max_steps):
            phase = "action_catalogue"
            actions = available_actions(env, args.action_space)
            phase = "policy"
            decision = policy.choose_action(observation, goal, actions)
            phase = "environment_step"
            next_observation, reward, done, info = env.step(decision["action"])
            score = float(info["score"])
            reward_sum += float(reward)
            result["steps"] = step + 1
            result["trajectory"].append({"step": step, "observation": observation, "decision": decision,
                                          "next_observation": next_observation, "reward": float(reward),
                                          "score": score, "done": bool(done), "valid": info.get("valid")})
            observation = next_observation
            phase = "feedback"
            policy.observe(decision["action"], next_observation, reward=float(reward), done=bool(done))
            if done:
                break
        result.update(status="completed", success=score >= 100.0, score=score, reward=reward_sum,
                      termination="environment_done" if done else "step_limit", environment_done=bool(done))
    except Exception as error:
        result.update(status="runtime_error", partial_score=score, partial_reward=reward_sum)
        result["errors"].append({"phase": phase, "type": type(error).__name__, "message": str(error)})
    result.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(),
                  model_calls=policy.calls, feedback=policy.feedback)
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [record for record in records if record["status"] == "completed"]
    count = len(complete)
    usage = {key: sum(record["usage"][key] for record in records)
             for key in ("calls", "errors", "prompt_tokens", "completion_tokens", "total_tokens", "elapsed_s", "retries", "provider_request_attempts", "provider_successful_calls")}
    costs = [record["usage"]["estimated_cost_usd"] for record in records]
    usage["estimated_cost_usd"] = sum(costs) if all(cost is not None for cost in costs) else None
    return {"benchmark": "scienceworld", "selected_tasks": len(records), "completed_tasks": count,
            "setup_errors": sum(record["status"] == "setup_error" for record in records),
            "runtime_errors": sum(record["status"] == "runtime_error" for record in records),
            "mean_score_completed": sum(record["score"] for record in complete) / count if count else None,
            "success_rate_completed": sum(record["success"] for record in complete) / count if count else None,
            "success_denominator": count, "all_selected_tasks_evaluated": count == len(records),
            "mean_steps_completed": sum(record["steps"] for record in complete) / count if count else None,
            "mean_calls_completed": sum(record["usage"]["calls"] for record in complete) / count if count else None,
            "mean_wall_time_s_completed": sum(record["wall_time_s"] for record in complete) / count if count else None,
            "usage_all_attempts": usage}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_policy_arguments(parser)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--jar-path", type=Path)
    parser.add_argument("--simplifications", default="")
    parser.add_argument("--action-space", choices=("possible", "valid"), default="possible")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--outdir", "--out-dir", dest="outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    tasks = load_tasks(args.task_manifest)
    prototype = policy_from_args(args)
    validate_evaluation_ids(prototype.graph, [task["task_id"] for task in tasks])
    if prototype.graph is not None:
        graph_benchmark = prototype.graph["provenance"].get("benchmark")
        if graph_benchmark is not None and str(graph_benchmark).lower().replace("-", "").replace("_", "") != "scienceworld":
            raise ValueError("Graph benchmark provenance does not match ScienceWorld.")
    run_dir = create_run_directory(args.outdir)
    metadata = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "task_manifest_sha256": file_sha256(args.task_manifest), "tasks": tasks, "source_api": SOURCE_API,
                "graph_sha256": file_sha256(args.graph) if args.graph else None,
                "protocol": "ScienceWorld text interaction; real simulator score, no gold path or hidden object tree"}
    write_json(run_dir / "configuration.json", metadata)
    env = None
    records = []
    try:
        from scienceworld import ScienceWorldEnv
        metadata["scienceworld_version"] = importlib.metadata.version("scienceworld")
        from scienceworld.constants import JAR_PATH
        metadata["simulator_jar_sha256"] = file_sha256(args.jar_path or JAR_PATH)
        env = ScienceWorldEnv("", str(args.jar_path) if args.jar_path else None, envStepLimit=args.max_steps + 1)
        validate_tasks(env, tasks, args)
        write_json(run_dir / "configuration.json", metadata)
    except Exception as error:
        write_json(run_dir / "summary.json", {"status": "setup_error", "selected_tasks": len(tasks),
                   "completed_tasks": 0, "error_type": type(error).__name__, "error": str(error)})
        if env is not None:
            env.close()
        raise
    try:
        with (run_dir / "episodes.jsonl").open("x", encoding="utf-8") as output:
            for task in tasks:
                record = evaluate_task(env, task, args)
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                write_json(run_dir / "summary.json", summarize(records))
        write_json(run_dir / "summary.json", summarize(records))
    finally:
        env.close()


if __name__ == "__main__":
    main()
