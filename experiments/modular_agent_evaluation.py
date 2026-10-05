from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.agent_evaluation_common import (
    add_policy_arguments,
    create_run_directory,
    file_sha256,
    policy_from_args,
    validate_evaluation_ids,
    write_json,
)


def select_indices(total, requested, limit):
    indices = list(requested) if requested is not None else list(range(total))[:limit]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Task indices must be nonempty and unique.")
    if any(index < 0 or index >= total for index in indices):
        raise ValueError(f"Task indices must be within [0, {total}).")
    return indices


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def summarize(records):
    complete = [record for record in records if record["status"] == "ok"]
    simulator_usage = [record["simulator_usage"] for record in complete if record.get("simulator_usage") is not None]
    return {
        "attempted": len(records),
        "evaluated": len(complete),
        "errors": len(records) - len(complete),
        "coverage": len(complete) / len(records) if records else None,
        "score": stats([record["score"] for record in complete]),
        "success_rate_pct": 100 * statistics.mean(record["success"] for record in complete) if complete else None,
        "steps": stats([record["steps"] for record in complete]),
        "agent_calls": stats([record["usage"]["calls"] for record in complete]),
        "agent_provider_request_attempts": stats([record["usage"]["provider_request_attempts"] for record in complete]),
        "agent_tokens": stats([record["usage"]["total_tokens"] for record in complete]),
        "episode_elapsed_s": stats([record["elapsed_s"] for record in complete]),
        "agent_cost_usd": stats([record["usage"]["estimated_cost_usd"] for record in complete
                                 if record["usage"]["estimated_cost_usd"] is not None]),
        "simulator_calls": stats([usage["calls"] for usage in simulator_usage]),
        "simulator_tokens": stats([usage["total_tokens"] for usage in simulator_usage]),
        "simulator_cost_usd": stats([usage["estimated_cost_usd"] for usage in simulator_usage
                                     if usage["estimated_cost_usd"] is not None]),
        "usage_all_attempts": {key: sum(record["usage"][key] for record in records)
                               for key in ("calls", "provider_request_attempts", "total_tokens", "errors")},
        "error_scoring": "Errors are reported separately; success and score denominators contain evaluated tasks only.",
    }


def select_alfworld(args):
    if args.alfworld_config is None or not args.alfworld_config.is_file():
        raise ValueError("ALFWorld requires --alfworld-config pointing to an installed dataset configuration.")
    import yaml
    from alfworld.agents.environment import get_environment

    with args.alfworld_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("ALFWorld configuration must be an object.")
    config = copy.deepcopy(config)
    config["dataset"]["num_eval_games"] = -1
    method = config["general"]["training_method"]
    section = {"dqn": "rl", "dagger": "dagger"}.get(method)
    if section is None:
        raise ValueError("ALFWorld supports dqn or dagger configuration layouts.")
    config[section]["training"]["max_nb_steps_per_episode"] = args.max_steps
    wrapper = get_environment("AlfredTWEnv")(config, train_eval=args.alfworld_split)
    games = sorted(set(wrapper.game_files))
    indices = select_indices(len(games), args.task_indices, args.n_tasks)
    data_key = "eval_ood_data_path" if args.alfworld_split == "eval_out_of_distribution" else "eval_id_data_path"
    data_root = Path(os.path.expandvars(config["dataset"][data_key])).expanduser().resolve()
    tasks = []
    for index in indices:
        path = Path(games[index]).resolve()
        if not path.is_file():
            raise ValueError(f"Missing ALFWorld game: {path}")
        relative = str(path.relative_to(data_root))
        tasks.append({"index": index, "task_id": f"alfworld:{args.alfworld_split}:{relative}",
                      "game_file": str(path), "game_sha256": file_sha256(path)})
    return wrapper, tasks


def select_tau(args):
    if args.benchmark == "tau_airline" and args.tau_split != "test":
        raise ValueError("The official airline benchmark exposes the test split only.")
    domain = "retail" if args.benchmark == "tau_retail" else "airline"
    module = importlib.import_module(f"tau_bench.envs.{domain}.tasks_{args.tau_split}")
    tasks = getattr(module, "TASKS" if domain == "airline" else f"TASKS_{args.tau_split.upper()}")
    env_module = importlib.import_module(f"tau_bench.envs.{domain}.env")
    env_class = getattr(env_module, "MockRetailDomainEnv" if domain == "retail" else "MockAirlineDomainEnv")
    indices = select_indices(len(tasks), args.task_indices, args.n_tasks)
    source_path = Path(module.__file__).resolve()
    selected = []
    for index in indices:
        item = tasks[index]
        task_content = item.model_dump(mode="json") if hasattr(item, "model_dump") else item.dict()
        task_bytes = json.dumps(task_content, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        selected.append({"index": index, "task_id": f"{args.benchmark}:{args.tau_split}:{index}",
                         "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
                         "task_source_file": str(source_path), "task_source_sha256": file_sha256(source_path),
                         "environment_source_sha256": file_sha256(Path(env_module.__file__).resolve())})
    return env_class, selected


def metered_tau_user(model, provider, seed, max_tokens, input_price, output_price):
    from litellm import completion
    from tau_bench.envs.user import LLMUserSimulationEnv

    class MeteredUser(LLMUserSimulationEnv):
        def __init__(self):
            self.model = model
            self.provider = provider
            self.messages = []
            self.calls = []
            self.total_cost = 0.0

        def generate_next_message(self, messages):
            started = time.perf_counter()
            record = {"status": "started", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self.calls.append(record)
            kwargs = {"model": self.model, "custom_llm_provider": self.provider,
                      "messages": messages, "temperature": 0.0, "seed": seed + len(self.calls) - 1,
                      "max_tokens": max_tokens}
            api_key = os.environ.get("FED_CAUSAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("FED_CAUSAL_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            if provider == "openai" and api_key:
                kwargs["api_key"] = api_key
            if provider == "openai" and base_url:
                kwargs["api_base"] = base_url
            try:
                response = completion(**kwargs)
                usage = getattr(response, "usage", None)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    record[key] = getattr(usage, key, 0) or 0
                message = response.choices[0].message
                content = message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("The benchmark user returned an empty response.")
                self.messages.append(message.model_dump())
                record["status"] = "ok"
                return content
            except Exception as error:
                record.update(status="error", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                record["elapsed_s"] = time.perf_counter() - started

        def snapshot_usage(self):
            usage = {key: sum(call.get(key, 0) for call in self.calls)
                     for key in ("prompt_tokens", "completion_tokens", "total_tokens", "elapsed_s")}
            usage.update(calls=len(self.calls), errors=sum(call["status"] == "error" for call in self.calls))
            usage["estimated_cost_usd"] = None if input_price is None else (
                usage["prompt_tokens"] * input_price + usage["completion_tokens"] * output_price) / 1000000
            usage["token_price_per_million"] = {"input": input_price, "output": output_price}
            return usage

        def get_total_cost(self):
            return self.snapshot_usage()["estimated_cost_usd"] or 0.0

    return MeteredUser()


def run_alfworld(args, wrapper, task, policy):
    env = None
    trajectory = []
    record = {"task_id": task["task_id"], "status": "error", "steps": 0, "score": None, "success": None}
    started = time.perf_counter()
    try:
        random.seed(args.seed)
        wrapper.game_files = [task["game_file"]]
        wrapper.num_games = 1
        env = wrapper.init_env(batch_size=1)
        if hasattr(env, "seed"):
            env.seed(args.seed)
        observations, infos = env.reset()
        observation = str(observations[0])
        goal = observation.split("Your task is to:", 1)[-1].strip()
        score, success, done = 0.0, False, False
        for step in range(args.max_steps):
            admissible = list(infos["admissible_commands"][0])
            decision = policy.choose_action(observation, goal, admissible)
            observations, rewards, dones, infos = env.step([decision["action"]])
            next_observation = str(observations[0])
            score, success, done = float(rewards[0]), bool(infos["won"][0]), bool(dones[0])
            trajectory.append({"step": step, "observation": observation, "available_actions": admissible,
                               "decision": decision, "next_observation": next_observation,
                               "reward": score, "won": success, "done": done})
            policy.observe(decision["action"], next_observation, reward=score, done=done)
            observation = next_observation
            if done:
                break
        record.update(status="ok", score=score, success=success, done=done,
                      termination="environment" if done else "step_limit")
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                record["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
    record.update(steps=len(trajectory), elapsed_s=time.perf_counter() - started,
                  usage=policy.snapshot_usage(), trajectory=trajectory, model_calls=policy.calls,
                  feedback=policy.feedback)
    return record


def run_tau(args, env_class, task, policy):
    from tau_bench.types import Action

    trajectory = []
    simulator = None
    record = {"task_id": task["task_id"], "status": "error", "steps": 0, "score": None, "success": None}
    started = time.perf_counter()
    try:
        random.seed(args.seed)
        env = env_class(user_strategy="human", task_split=args.tau_split, task_index=task["index"])
        simulator = metered_tau_user(args.user_model or args.model, args.user_provider, args.seed,
                                     args.user_max_tokens, args.user_input_price_per_million,
                                     args.user_output_price_per_million)
        env.user = simulator
        reset = env.reset(task_index=task["index"])
        observation = reset.observation
        goal = "Resolve the customer's request according to the domain policy. Initial customer message: " + observation
        public_policy = {"wiki": env.wiki, "rules": env.rules}
        tools = list(env.tools_info) + [{"type": "function", "function": {
            "name": "respond", "description": "Send a message to the customer.", "parameters": {
                "type": "object", "properties": {"content": {"type": "string"}},
                "required": ["content"], "additionalProperties": False}}}]
        score, done = 0.0, False
        for step in range(args.max_steps):
            context = {"message": observation, "domain_policy": public_policy}
            decision = policy.choose_action(context, goal, tools)
            action = decision["action"]
            response = env.step(Action(name=action["name"], kwargs=action["arguments"]))
            score, done = float(response.reward), bool(response.done)
            trajectory.append({"step": step, "observation": observation, "decision": decision,
                               "next_observation": response.observation, "reward": score, "done": done})
            policy.observe(action, response.observation, reward=score, done=done)
            observation = response.observation
            if done:
                break
        record.update(status="ok", score=score, success=score == 1.0, done=done,
                      termination="environment" if done else "step_limit",
                      scoring="Official terminal reward; step-limit episodes retain the last environment reward.")
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    record.update(steps=len(trajectory), elapsed_s=time.perf_counter() - started,
                  usage=policy.snapshot_usage(), trajectory=trajectory, model_calls=policy.calls,
                  feedback=policy.feedback, simulator_usage=simulator.snapshot_usage() if simulator else None,
                  simulator_model=args.user_model or args.model, simulator_provider=args.user_provider)
    return record


def main():
    parser = argparse.ArgumentParser(description="Run graph, attention-anchor, control, feedback, and model comparisons on installed ALFWorld or tau-bench tasks.")
    add_policy_arguments(parser)
    parser.add_argument("--benchmark", choices=("alfworld", "tau_retail", "tau_airline"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-indices", type=int, nargs="+")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--alfworld-config", type=Path)
    parser.add_argument("--alfworld-split", choices=("eval_in_distribution", "eval_out_of_distribution"), default="eval_out_of_distribution")
    parser.add_argument("--tau-split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--user-model")
    parser.add_argument("--user-provider", default="openai")
    parser.add_argument("--user-max-tokens", type=int, default=1024)
    parser.add_argument("--user-input-price-per-million", type=float)
    parser.add_argument("--user-output-price-per-million", type=float)
    args = parser.parse_args()
    if min(args.n_tasks, args.max_steps, args.user_max_tokens) < 1:
        parser.error("Task count, step limit, and user token budget must be positive.")
    seeds, models = args.seeds or [args.seed], args.models or [args.model]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds) or len(set(models)) != len(models) or any(not model for model in models):
        parser.error("Seeds and models must be nonempty, unique, and valid.")
    user_prices = (args.user_input_price_per_million, args.user_output_price_per_million)
    if (user_prices[0] is None) != (user_prices[1] is None) or any(price is not None and (not math.isfinite(price) or price < 0) for price in user_prices):
        parser.error("Supply both nonnegative finite simulator token prices or neither.")
    prototype = policy_from_args(args)
    adapter, tasks = select_alfworld(args) if args.benchmark == "alfworld" else select_tau(args)
    validate_evaluation_ids(prototype.graph, [task["task_id"] for task in tasks])
    if prototype.graph is not None:
        benchmark = prototype.graph["provenance"].get("benchmark")
        if benchmark and benchmark != args.benchmark:
            parser.error("Graph provenance benchmark does not match the selected benchmark.")
    if args.benchmark != "alfworld":
        importlib.import_module("litellm")
    versions = {}
    for package in ("alfworld", "tau-bench", "openai", "litellm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    out_dir = create_run_directory(args.out_dir)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {"benchmark": args.benchmark, "protocol": "supplied-interface modular LLM policy evaluation",
                "config": config, "models": models, "seeds": seeds, "tasks": tasks, "versions": versions,
                "graph_sha256": file_sha256(args.graph) if args.graph else None,
                "graph_provenance": prototype.graph["provenance"] if prototype.graph else None,
                "alfworld_config_sha256": file_sha256(args.alfworld_config) if args.alfworld_config else None,
                "mechanism_backend": "LLM predictions conditioned on supplied module descriptions and interface evidence",
                "seed_scope": "Environment RNG and requested model seeds; API determinism is provider-dependent.",
                "simulator_model": (args.user_model or args.model) if args.benchmark != "alfworld" else None,
                "simulator_provider": args.user_provider if args.benchmark != "alfworld" else None,
                "cost_scope": "Agent and benchmark-user usage are separate; costs require explicit token prices."}
    write_json(out_dir / "manifest.json", manifest)
    records = []
    for model_index, model in enumerate(models):
        for seed in seeds:
            current = argparse.Namespace(**vars(args))
            current.model, current.seed = model, seed
            current.user_model = args.user_model or args.model
            for task in tasks:
                policy = policy_from_args(current)
                record = run_alfworld(current, adapter, task, policy) if args.benchmark == "alfworld" else run_tau(current, adapter, task, policy)
                record.update(benchmark=args.benchmark, model=model, seed=seed, condition=args.condition,
                              attention_anchor=args.attention_anchor, immediate_feedback=args.immediate_feedback,
                              control=policy.control, graph_sha256=manifest["graph_sha256"])
                filename = f"episode_m{model_index}_s{seed}_t{task['index']}.json"
                write_json(out_dir / filename, record)
                records.append({key: value for key, value in record.items() if key not in ("trajectory", "model_calls", "feedback")})
                write_json(out_dir / "records.json", records)
                groups = [{"model": selected_model, "seed": selected_seed,
                           **summarize([item for item in records if item["model"] == selected_model and item["seed"] == selected_seed])}
                          for selected_model in models for selected_seed in seeds]
                write_json(out_dir / "summary.json", {"overall": summarize(records), "groups": groups})
    print(json.dumps({"out_dir": str(out_dir), **summarize(records)}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
