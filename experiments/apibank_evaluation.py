from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import random
from pathlib import Path
import sys
import time
import textwrap
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.agent_evaluation_common import add_policy_arguments, create_run_directory, file_sha256
from experiments.agent_evaluation_common import policy_from_args, validate_evaluation_ids, write_json


SOURCE_API = "https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank"
PROTOCOL = "stateful_dialogue_fixed_call_slots"


def load_dialogues(data_dir: Path, manifest_path: Path | None) -> list[dict[str, Any]]:
    data_dir = data_dir.resolve(strict=True)
    if manifest_path is None:
        paths = sorted(data_dir.rglob("*.jsonl"))
    else:
        with manifest_path.open(encoding="utf-8") as handle:
            relative_paths = json.load(handle)
        if not isinstance(relative_paths, list) or not relative_paths or not all(isinstance(item, str) for item in relative_paths):
            raise ValueError("Dialogue manifest must be a nonempty JSON list of relative JSONL paths.")
        paths = [(data_dir / item).resolve(strict=True) for item in relative_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("No dialogue files selected, or duplicate files selected.")
    tasks = []
    for path in paths:
        if not path.is_relative_to(data_dir) or path.suffix != ".jsonl":
            raise ValueError("Dialogue paths must be JSONL files inside --data-dir.")
        with path.open(encoding="utf-8") as handle:
            turns = [json.loads(line) for line in handle if line.strip()]
        has_user = False
        calls = 0
        for turn in turns:
            if not isinstance(turn, dict) or turn.get("role") not in ("User", "AI", "API"):
                raise ValueError(f"Malformed dialogue turn in {path}.")
            if turn["role"] in ("User", "AI"):
                if not isinstance(turn.get("text"), str):
                    raise ValueError(f"Text is required for dialogue turns in {path}.")
                has_user |= turn["role"] == "User"
            else:
                if not has_user or not isinstance(turn.get("api_name"), str) or not isinstance(turn.get("param_dict"), dict):
                    raise ValueError(f"API turn lacks an observed user goal or valid API fields in {path}.")
                result = turn.get("result")
                if not isinstance(result, dict) or "output" not in result or "exception" not in result:
                    raise ValueError(f"API turn lacks an official reference result in {path}.")
                calls += 1
        if not calls:
            raise ValueError(f"No API calls in selected dialogue: {path}")
        relative_path = path.relative_to(data_dir).as_posix()
        tasks.append({"task_id": "apibank:" + relative_path, "relative_path": relative_path,
                      "source_sha256": file_sha256(path), "turns": turns, "total_calls": calls})
    return tasks


def load_tool_manager(benchmark_dir: Path) -> Any:
    expected = (benchmark_dir / "tool_manager.py").resolve(strict=True)
    if not (benchmark_dir / "apis").is_dir() or not (benchmark_dir / "init_database").is_dir():
        raise ValueError("--benchmark-dir must point to the upstream api-bank directory.")
    sys.path.insert(0, str(benchmark_dir))
    if "apis" in sys.modules:
        origin = Path(sys.modules["apis"].__file__).resolve()
        if not origin.is_relative_to(benchmark_dir):
            raise RuntimeError("Another package named apis is already imported; run this evaluator in a fresh process.")
    specification = importlib.util.spec_from_file_location("fed_causal_apibank_tool_manager", expected)
    if specification is None or specification.loader is None:
        raise ImportError("Cannot load the upstream APIBank ToolManager.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.ToolManager


def required_parameters(manager: Any, name: str) -> list[str]:
    tool_class = manager.get_api_by_name(name)["class"]
    signature = inspect.signature(tool_class.call)
    return [parameter.name for parameter in signature.parameters.values()
            if parameter.name != "self" and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and parameter.default is inspect.Parameter.empty]


def tool_catalogue(manager: Any) -> list[dict[str, Any]]:
    catalogue = []
    type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
                "list": "array", "list(str)": "array"}
    for name in sorted(manager.list_all_apis()):
        description = json.loads(manager.get_api_description(name))
        properties = {}
        for key, parameter in description["input_parameters"].items():
            item = {"description": parameter.get("description", "")}
            if parameter["type"] in type_map:
                item["type"] = type_map[parameter["type"]]
            else:
                item["description"] += f" Declared upstream type: {parameter['type']}."
            if parameter["type"] == "list(str)":
                item["items"] = {"type": "string"}
            properties[key] = item
        catalogue.append({"type": "function", "function": {
            "name": name, "description": description["description"],
            "parameters": {"type": "object", "properties": properties,
                           "required": required_parameters(manager, name), "additionalProperties": False}}})
    if not catalogue:
        raise ValueError("The upstream API catalogue is empty.")
    return catalogue


def validate_tasks(manager: Any, tasks: list[dict[str, Any]]) -> None:
    names = set(manager.list_all_apis())
    for task in tasks:
        for turn in task["turns"]:
            if turn["role"] != "API":
                continue
            if turn["api_name"] not in names:
                raise ValueError(f"Unknown reference tool {turn['api_name']} in {task['task_id']}")
            parameter_names = set(manager.get_api_by_name(turn["api_name"])["input_parameters"])
            provided_parameters = set(turn["param_dict"])
            required = set(required_parameters(manager, turn["api_name"]))
            if not provided_parameters.issubset(parameter_names) or not required.issubset(provided_parameters):
                raise ValueError(f"Reference parameter schema mismatch in {task['task_id']}:{turn['api_name']}")
            tool = manager.init_tool(turn["api_name"])
            if not callable(getattr(tool, "check_api_call_correctness", None)):
                raise ValueError(f"Upstream correctness checker missing for {turn['api_name']}")


def stochastic_reference_tools(manager: Any) -> dict[str, list[str]]:
    result = {}
    ignored_random_calls = {"seed", "getstate", "setstate"}
    for name in sorted(manager.list_all_apis()):
        tool_class = manager.get_api_by_name(name)["class"]
        module_path = inspect.getsourcefile(tool_class)
        if module_path is None:
            raise ValueError(f"Cannot inspect reference compatibility for upstream API {name}.")
        module_tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
        aliases = {}
        direct = {}
        for node in ast.walk(module_tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name in ("random", "uuid", "secrets"):
                        aliases[imported.asname or imported.name] = imported.name
            elif isinstance(node, ast.ImportFrom) and node.module in ("random", "uuid", "secrets"):
                for imported in node.names:
                    direct[imported.asname or imported.name] = (node.module, imported.name)
        class_tree = ast.parse(textwrap.dedent(inspect.getsource(tool_class)))
        generators = set()
        for node in ast.walk(class_tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            origin = None
            if isinstance(target, ast.Name):
                origin = direct.get(target.id)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in aliases:
                origin = (aliases[target.value.id], target.attr)
            if origin is not None and not (origin[0] == "random" and origin[1] in ignored_random_calls):
                generators.add(".".join(origin))
        if generators:
            result[name] = sorted(generators)
    return result


def execute_tool(manager: Any, action: dict[str, Any]) -> Any:
    parameters = dict(action["arguments"])
    specification = manager.get_api_by_name(action["name"])["input_parameters"]
    for name, value in parameters.items():
        if specification[name]["type"] == "bool" and isinstance(value, bool):
            parameters[name] = "True" if value else "False"
    return manager.api_call(action["name"], **parameters)


def evaluate_dialogue(manager_type: Any, catalogue: list[dict[str, Any]], task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    policy = policy_from_args(args)
    record = {"benchmark": "apibank", "protocol": PROTOCOL, "task_id": task["task_id"],
              "source_sha256": task["source_sha256"], "seed": args.seed, "condition": args.condition,
              "status": "setup_error", "success": None, "reward": None, "call_accuracy": None,
              "correct_calls": 0, "total_calls": task["total_calls"], "steps": 0, "trajectory": [], "errors": []}
    incompatible = task.get("incompatible_reference_apis", {})
    if incompatible:
        record["status"] = "protocol_incompatible"
        record["errors"].append({"phase": "protocol_validation", "type": "StochasticReferenceMismatch",
            "message": "Upstream APIs generate stochastic values, so saved-reference exact result checks are not validated for persistent regenerated state. No calls were executed and no gold identifiers were injected.",
            "apis": incompatible})
        record.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(), model_calls=[], feedback=[])
        return record
    phase = "environment_setup"
    environment_seed = (args.seed + int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:16], 16)) % (2 ** 32)
    record["environment_seed"] = environment_seed
    previous_random_state = random.getstate()
    random.seed(environment_seed)
    try:
        manager = manager_type()
        history = []
        user_turns = []
        record["status"] = "running"
        for turn in task["turns"]:
            if turn["role"] == "AI":
                continue
            if turn["role"] == "User":
                user_turns.append(turn["text"])
                history.append({"role": "user", "content": turn["text"]})
                continue
            phase = "policy"
            goal = "Complete the user's currently observed requests through the available API tools. The benchmark reference year is 2023.\n" + "\n".join(user_turns)
            observation = {"observed_conversation": history}
            decision = policy.choose_action(observation, goal, catalogue, history=history)
            action = decision["action"]
            phase = "tool_execution"
            result = execute_tool(manager, action)
            record["steps"] += 1
            history.append({"role": "assistant", "tool_call": action})
            history.append({"role": "tool", "name": action["name"], "content": result})
            entry = {"call_index": record["steps"] - 1, "decision": decision, "tool_result": result,
                     "correct": None, "reference_api": turn["api_name"]}
            record["trajectory"].append(entry)
            phase = "official_scoring"
            if action["name"] == turn["api_name"]:
                checker = manager.init_tool(action["name"])
                correct = checker.check_api_call_correctness(copy.deepcopy(result), copy.deepcopy(turn["result"]))
                if not isinstance(correct, bool):
                    raise ValueError("Official APIBank checker did not return a boolean.")
            else:
                correct = False
            entry["correct"] = correct
            record["correct_calls"] += int(correct)
            phase = "feedback"
            policy.observe(action, result, reward=None, done=record["steps"] == task["total_calls"])
        record.update(status="completed", success=record["correct_calls"] == task["total_calls"],
                      call_accuracy=record["correct_calls"] / task["total_calls"],
                      reward=record["correct_calls"] / task["total_calls"])
    except Exception as error:
        record["status"] = "setup_error" if phase == "environment_setup" else "runtime_error"
        record["errors"].append({"phase": phase, "type": type(error).__name__, "message": str(error)})
    random.setstate(previous_random_state)
    record.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(),
                  model_calls=policy.calls, feedback=policy.feedback)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [record for record in records if record["status"] == "completed"]
    count = len(complete)
    denominator = sum(record["total_calls"] for record in complete)
    usage = {key: sum(record["usage"][key] for record in records)
             for key in ("calls", "errors", "prompt_tokens", "completion_tokens", "total_tokens", "elapsed_s",
                         "retries", "provider_request_attempts", "provider_successful_calls")}
    costs = [record["usage"]["estimated_cost_usd"] for record in records]
    usage["estimated_cost_usd"] = sum(costs) if all(cost is not None for cost in costs) else None
    return {"benchmark": "apibank", "protocol": PROTOCOL, "selected_dialogues": len(records),
            "completed_dialogues": count, "setup_errors": sum(record["status"] == "setup_error" for record in records),
            "runtime_errors": sum(record["status"] == "runtime_error" for record in records),
            "protocol_incompatible_dialogues": sum(record["status"] == "protocol_incompatible" for record in records),
            "all_calls_correct_rate_completed": sum(record["success"] for record in complete) / count if count else None,
            "micro_call_accuracy_completed": sum(record["correct_calls"] for record in complete) / denominator if denominator else None,
            "macro_call_accuracy_completed": sum(record["call_accuracy"] for record in complete) / count if count else None,
            "call_accuracy_denominator": denominator, "dialogue_success_denominator": count,
            "all_selected_dialogues_evaluated": count == len(records),
            "mean_calls_completed": sum(record["usage"]["calls"] for record in complete) / count if count else None,
            "mean_wall_time_s_completed": sum(record["wall_time_s"] for record in complete) / count if count else None,
            "usage_all_attempts": usage}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_policy_arguments(parser)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dialogue-manifest", type=Path)
    parser.add_argument("--outdir", "--out-dir", dest="outdir", type=Path, required=True)
    args = parser.parse_args()
    args.benchmark_dir = args.benchmark_dir.expanduser().resolve(strict=True)
    args.data_dir = args.data_dir.expanduser().resolve(strict=True)
    args.outdir = args.outdir.expanduser().resolve()
    if args.graph:
        args.graph = args.graph.expanduser().resolve(strict=True)
    if args.dialogue_manifest:
        args.dialogue_manifest = args.dialogue_manifest.expanduser().resolve(strict=True)
    tasks = load_dialogues(args.data_dir, args.dialogue_manifest)
    prototype = policy_from_args(args)
    validate_evaluation_ids(prototype.graph, [task["task_id"] for task in tasks])
    if prototype.graph is not None:
        graph_benchmark = prototype.graph["provenance"].get("benchmark")
        if graph_benchmark is not None and str(graph_benchmark).lower().replace("-", "").replace("_", "") not in ("apibank", "apibankstateful"):
            raise ValueError("Graph benchmark provenance does not match APIBank.")
    run_dir = create_run_directory(args.outdir)
    selected = [{key: value for key, value in task.items() if key != "turns"} for task in tasks]
    digest_payload = {task["relative_path"]: task["source_sha256"] for task in tasks}
    metadata = {"benchmark": "apibank", "protocol": PROTOCOL, "source_api": SOURCE_API,
                "protocol_description": "Persistent upstream ToolManager state per dialogue; replay only observed user turns; choose one tool at each reference API slot; score with upstream API correctness checkers. Reference AI text, target tools, parameters, and results are never policy inputs. This is a fixed-call-slot diagnostic, not a free-dialogue end-to-end benchmark.",
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "tasks": selected, "dataset_sha256": hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest(),
                "graph_sha256": file_sha256(args.graph) if args.graph else None,
                "environment_seed_scope": "Python random reset per dialogue to (seed + SHA256(task_id) first 64 bits) modulo 2^32; separate from policy request seed",
                "parameter_schema_policy": "Required parameters derived from upstream call signatures; unsupported JSON-schema type names preserve upstream descriptions and are checked by ToolManager at execution"}
    write_json(run_dir / "configuration.json", metadata)
    previous_directory = Path.cwd()
    records = []
    try:
        os.chdir(args.benchmark_dir)
        try:
            manager_type = load_tool_manager(args.benchmark_dir)
            manager = manager_type()
            catalogue = tool_catalogue(manager)
            validate_tasks(manager, tasks)
            stochastic = stochastic_reference_tools(manager)
            for task in tasks:
                task["incompatible_reference_apis"] = {turn["api_name"]: stochastic[turn["api_name"]]
                    for turn in task["turns"] if turn["role"] == "API" and turn["api_name"] in stochastic}
            metadata["tasks"] = [{key: value for key, value in task.items() if key != "turns"} for task in tasks]
            metadata["reference_compatibility_policy"] = "Conservatively exclude dialogues whose APIs contain executable random/UUID/secrets generation; report protocol_incompatible without injecting saved identifiers or changing upstream checkers."
            metadata["tool_manager_sha256"] = file_sha256(args.benchmark_dir / "tool_manager.py")
            metadata["api_source_sha256"] = {path.name: file_sha256(path) for path in sorted((args.benchmark_dir / "apis").glob("*.py"))}
            metadata["initial_database_sha256"] = {path.name: file_sha256(path) for path in sorted((args.benchmark_dir / "init_database").glob("*.json"))}
            metadata["tool_catalogue"] = catalogue
            write_json(run_dir / "configuration.json", metadata)
        except Exception as error:
            write_json(run_dir / "summary.json", {"status": "setup_error", "selected_dialogues": len(tasks),
                       "completed_dialogues": 0, "error_type": type(error).__name__, "error": str(error)})
            raise
        with (run_dir / "episodes.jsonl").open("x", encoding="utf-8") as output:
            for task in tasks:
                record = evaluate_dialogue(manager_type, catalogue, task, args)
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                write_json(run_dir / "summary.json", summarize(records))
        write_json(run_dir / "summary.json", summarize(records))
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
