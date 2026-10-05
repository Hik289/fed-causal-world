from __future__ import annotations

import argparse
import ast
import copy
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import textwrap
import time
import traceback
from typing import Any, Dict, List

AZURE_API_KEY = os.environ.get("FED_CAUSAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
AZURE_API_BASE = os.environ.get("FED_CAUSAL_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
MODEL_NAME = os.environ.get("FED_CAUSAL_MODEL", "openai/gpt-5.4-mini")
DEPLOYMENT = MODEL_NAME.rsplit("/", 1)[-1]

PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00



_original_completion = None
_usage_log: List[Dict[str, Any]] = []


def _patched_completion(*args, **kwargs):
    if AZURE_API_KEY:
        kwargs.setdefault("api_key", AZURE_API_KEY)
    if AZURE_API_BASE:
        kwargs.setdefault("api_base", AZURE_API_BASE)
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


def _enable_legacy_api_metering():
    global _original_completion
    import litellm
    if litellm.completion is not _patched_completion:
        _original_completion = litellm.completion
        litellm.completion = _patched_completion



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




@lru_cache(maxsize=1)
def _legacy_agent_type():
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent

    class PromptHeaderAgent(ToolCallingAgent):
        def __init__(self, header: str, *args, **kwargs):
            self._baseline_header = header
            super().__init__(*args, **kwargs)
            self.wiki = f"{header}\n\n---\n\n{self.wiki}"

    return PromptHeaderAgent


def __getattr__(name):
    if name == "PromptHeaderAgent":
        return _legacy_agent_type()
    if name == "MockRetailDomainEnv":
        from tau_bench.envs.retail.env import MockRetailDomainEnv
        return MockRetailDomainEnv
    if name == "ToolCallingAgent":
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        return ToolCallingAgent
    if name == "UserStrategy":
        from tau_bench.envs.user import UserStrategy
        return UserStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



def run_one_baseline(baseline_id: str, task_indices: List[int],
                     log_dir: str) -> Dict[str, Any]:
    from tau_bench.envs.retail.env import MockRetailDomainEnv
    from tau_bench.envs.user import UserStrategy
    PromptHeaderAgent = _legacy_agent_type()
    _enable_legacy_api_metering()
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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+",
                    default=["B0_NoWM", "B1_LocalWM", "B2_GlobalSeqWM",
                             "B3_TransitionGraph", "B4_CorrelationWM",
                             "B5_TemporalWM", "B7_CausalWMNoInt",
                             "B8_FedCausalCompose", "B10_OracleCausalWM"])
    ap.add_argument("--task_indices", type=int, nargs="*",
                    default=list(range(21)))
    ap.add_argument("--log_dir", default="runs/tau_bench")
    ap.add_argument("--cell_cost_cap", type=float, default=30.0,
                    help="Per-baseline cost cap; if exceeded, abort and report")
    args = ap.parse_args(argv)
    _enable_legacy_api_metering()

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




def _load_evaluation_helpers():
    global add_policy_arguments, create_run_directory, file_sha256, policy_from_args, validate_evaluation_ids, write_json
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from fed_causal.llm_client import add_policy_arguments, create_run_directory, file_sha256
    from fed_causal.llm_client import policy_from_args, validate_evaluation_ids, write_json


def modular_select_indices(total, requested, limit):
    indices = list(requested) if requested is not None else list(range(total))[:limit]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Task indices must be nonempty and unique.')
    if any((index < 0 or index >= total for index in indices)):
        raise ValueError(f'Task indices must be within [0, {total}).')
    return indices

def modular_stats(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None, 'std': statistics.stdev(values) if len(values) > 1 else None}

def modular_summarize(records):
    complete = [record for record in records if record['status'] == 'ok']
    simulator_usage = [record['simulator_usage'] for record in complete if record.get('simulator_usage') is not None]
    return {'attempted': len(records), 'evaluated': len(complete), 'errors': len(records) - len(complete), 'coverage': len(complete) / len(records) if records else None, 'score': modular_stats([record['score'] for record in complete]), 'success_rate_pct': 100 * statistics.mean((record['success'] for record in complete)) if complete else None, 'steps': modular_stats([record['steps'] for record in complete]), 'agent_calls': modular_stats([record['usage']['calls'] for record in complete]), 'agent_provider_request_attempts': modular_stats([record['usage']['provider_request_attempts'] for record in complete]), 'agent_tokens': modular_stats([record['usage']['total_tokens'] for record in complete]), 'episode_elapsed_s': modular_stats([record['elapsed_s'] for record in complete]), 'agent_cost_usd': modular_stats([record['usage']['estimated_cost_usd'] for record in complete if record['usage']['estimated_cost_usd'] is not None]), 'simulator_calls': modular_stats([usage['calls'] for usage in simulator_usage]), 'simulator_tokens': modular_stats([usage['total_tokens'] for usage in simulator_usage]), 'simulator_cost_usd': modular_stats([usage['estimated_cost_usd'] for usage in simulator_usage if usage['estimated_cost_usd'] is not None]), 'usage_all_attempts': {key: sum((record['usage'][key] for record in records)) for key in ('calls', 'provider_request_attempts', 'total_tokens', 'errors')}, 'error_scoring': 'Errors are reported separately; success and score denominators contain evaluated tasks only.'}

def modular_select_alfworld(args):
    if args.alfworld_config is None or not args.alfworld_config.is_file():
        raise ValueError('ALFWorld requires --alfworld-config pointing to an installed dataset configuration.')
    import yaml
    from alfworld.agents.environment import get_environment
    with args.alfworld_config.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('ALFWorld configuration must be an object.')
    config = copy.deepcopy(config)
    config['dataset']['num_eval_games'] = -1
    method = config['general']['training_method']
    section = {'dqn': 'rl', 'dagger': 'dagger'}.get(method)
    if section is None:
        raise ValueError('ALFWorld supports dqn or dagger configuration layouts.')
    config[section]['training']['max_nb_steps_per_episode'] = args.max_steps
    wrapper = get_environment('AlfredTWEnv')(config, train_eval=args.alfworld_split)
    games = sorted(set(wrapper.game_files))
    indices = modular_select_indices(len(games), args.task_indices, args.n_tasks)
    data_key = 'eval_ood_data_path' if args.alfworld_split == 'eval_out_of_distribution' else 'eval_id_data_path'
    data_root = Path(os.path.expandvars(config['dataset'][data_key])).expanduser().resolve()
    tasks = []
    for index in indices:
        path = Path(games[index]).resolve()
        if not path.is_file():
            raise ValueError(f'Missing ALFWorld game: {path}')
        relative = str(path.relative_to(data_root))
        tasks.append({'index': index, 'task_id': f'alfworld:{args.alfworld_split}:{relative}', 'game_file': str(path), 'game_sha256': file_sha256(path)})
    return (wrapper, tasks)

def modular_select_tau(args):
    if args.benchmark == 'tau_airline' and args.tau_split != 'test':
        raise ValueError('The official airline benchmark exposes the test split only.')
    domain = 'retail' if args.benchmark == 'tau_retail' else 'airline'
    module = importlib.import_module(f'tau_bench.envs.{domain}.tasks_{args.tau_split}')
    tasks = getattr(module, 'TASKS' if domain == 'airline' else f'TASKS_{args.tau_split.upper()}')
    env_module = importlib.import_module(f'tau_bench.envs.{domain}.env')
    env_class = getattr(env_module, 'MockRetailDomainEnv' if domain == 'retail' else 'MockAirlineDomainEnv')
    indices = modular_select_indices(len(tasks), args.task_indices, args.n_tasks)
    source_path = Path(module.__file__).resolve()
    selected = []
    for index in indices:
        item = tasks[index]
        task_content = item.model_dump(mode='json') if hasattr(item, 'model_dump') else item.dict()
        task_bytes = json.dumps(task_content, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        selected.append({'index': index, 'task_id': f'{args.benchmark}:{args.tau_split}:{index}', 'task_sha256': hashlib.sha256(task_bytes).hexdigest(), 'task_source_file': str(source_path), 'task_source_sha256': file_sha256(source_path), 'environment_source_sha256': file_sha256(Path(env_module.__file__).resolve())})
    return (env_class, selected)

def modular_metered_tau_user(model, provider, seed, max_tokens, input_price, output_price):
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
            record = {'status': 'started', 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            self.calls.append(record)
            kwargs = {'model': self.model, 'custom_llm_provider': self.provider, 'messages': messages, 'temperature': 0.0, 'seed': seed + len(self.calls) - 1, 'max_tokens': max_tokens}
            api_key = os.environ.get('FED_CAUSAL_API_KEY') or os.environ.get('OPENAI_API_KEY')
            base_url = os.environ.get('FED_CAUSAL_API_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
            if provider == 'openai' and api_key:
                kwargs['api_key'] = api_key
            if provider == 'openai' and base_url:
                kwargs['api_base'] = base_url
            try:
                response = completion(**kwargs)
                usage = getattr(response, 'usage', None)
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    record[key] = getattr(usage, key, 0) or 0
                message = response.choices[0].message
                content = message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('The benchmark user returned an empty response.')
                self.messages.append(message.model_dump())
                record['status'] = 'ok'
                return content
            except Exception as error:
                record.update(status='error', error_type=type(error).__name__, error=str(error))
                raise
            finally:
                record['elapsed_s'] = time.perf_counter() - started

        def snapshot_usage(self):
            usage = {key: sum((call.get(key, 0) for call in self.calls)) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'elapsed_s')}
            usage.update(calls=len(self.calls), errors=sum((call['status'] == 'error' for call in self.calls)))
            usage['estimated_cost_usd'] = None if input_price is None else (usage['prompt_tokens'] * input_price + usage['completion_tokens'] * output_price) / 1000000
            usage['token_price_per_million'] = {'input': input_price, 'output': output_price}
            return usage

        def get_total_cost(self):
            return self.snapshot_usage()['estimated_cost_usd'] or 0.0
    return MeteredUser()

def modular_run_alfworld(args, wrapper, task, policy):
    env = None
    trajectory = []
    record = {'task_id': task['task_id'], 'status': 'error', 'steps': 0, 'score': None, 'success': None}
    started = time.perf_counter()
    try:
        random.seed(args.seed)
        wrapper.game_files = [task['game_file']]
        wrapper.num_games = 1
        env = wrapper.init_env(batch_size=1)
        if hasattr(env, 'seed'):
            env.seed(args.seed)
        (observations, infos) = env.reset()
        observation = str(observations[0])
        goal = observation.split('Your task is to:', 1)[-1].strip()
        (score, success, done) = (0.0, False, False)
        for step in range(args.max_steps):
            admissible = list(infos['admissible_commands'][0])
            decision = policy.choose_action(observation, goal, admissible)
            (observations, rewards, dones, infos) = env.step([decision['action']])
            next_observation = str(observations[0])
            (score, success, done) = (float(rewards[0]), bool(infos['won'][0]), bool(dones[0]))
            trajectory.append({'step': step, 'observation': observation, 'available_actions': admissible, 'decision': decision, 'next_observation': next_observation, 'reward': score, 'won': success, 'done': done})
            policy.observe(decision['action'], next_observation, reward=score, done=done)
            observation = next_observation
            if done:
                break
        record.update(status='ok', score=score, success=success, done=done, termination='environment' if done else 'step_limit')
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                record['cleanup_error'] = {'type': type(error).__name__, 'message': str(error)}
    record.update(steps=len(trajectory), elapsed_s=time.perf_counter() - started, usage=policy.snapshot_usage(), trajectory=trajectory, model_calls=policy.calls, feedback=policy.feedback)
    return record

def modular_run_tau(args, env_class, task, policy):
    from tau_bench.types import Action
    trajectory = []
    simulator = None
    record = {'task_id': task['task_id'], 'status': 'error', 'steps': 0, 'score': None, 'success': None}
    started = time.perf_counter()
    try:
        random.seed(args.seed)
        env = env_class(user_strategy='human', task_split=args.tau_split, task_index=task['index'])
        simulator = modular_metered_tau_user(args.user_model or args.model, args.user_provider, args.seed, args.user_max_tokens, args.user_input_price_per_million, args.user_output_price_per_million)
        env.user = simulator
        reset = env.reset(task_index=task['index'])
        observation = reset.observation
        goal = "Resolve the customer's request according to the domain policy. Initial customer message: " + observation
        public_policy = {'wiki': env.wiki, 'rules': env.rules}
        tools = list(env.tools_info) + [{'type': 'function', 'function': {'name': 'respond', 'description': 'Send a message to the customer.', 'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content'], 'additionalProperties': False}}}]
        (score, done) = (0.0, False)
        for step in range(args.max_steps):
            context = {'message': observation, 'domain_policy': public_policy}
            decision = policy.choose_action(context, goal, tools)
            action = decision['action']
            response = env.step(Action(name=action['name'], kwargs=action['arguments']))
            (score, done) = (float(response.reward), bool(response.done))
            trajectory.append({'step': step, 'observation': observation, 'decision': decision, 'next_observation': response.observation, 'reward': score, 'done': done})
            policy.observe(action, response.observation, reward=score, done=done)
            observation = response.observation
            if done:
                break
        record.update(status='ok', score=score, success=score == 1.0, done=done, termination='environment' if done else 'step_limit', scoring='Official terminal reward; step-limit episodes retain the last environment reward.')
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    record.update(steps=len(trajectory), elapsed_s=time.perf_counter() - started, usage=policy.snapshot_usage(), trajectory=trajectory, model_calls=policy.calls, feedback=policy.feedback, simulator_usage=simulator.snapshot_usage() if simulator else None, simulator_model=args.user_model or args.model, simulator_provider=args.user_provider)
    return record

def modular_evaluation_main(argv=None):
    _load_evaluation_helpers()
    parser = argparse.ArgumentParser(description='Run graph, attention-anchor, control, feedback, and model comparisons on installed ALFWorld or tau-bench tasks.')
    add_policy_arguments(parser)
    parser.add_argument('--benchmark', choices=('alfworld', 'tau_retail', 'tau_airline'), required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--task-indices', type=int, nargs='+')
    parser.add_argument('--n-tasks', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--alfworld-config', type=Path)
    parser.add_argument('--alfworld-split', choices=('eval_in_distribution', 'eval_out_of_distribution'), default='eval_out_of_distribution')
    parser.add_argument('--tau-split', choices=('train', 'dev', 'test'), default='test')
    parser.add_argument('--user-model')
    parser.add_argument('--user-provider', default='openai')
    parser.add_argument('--user-max-tokens', type=int, default=1024)
    parser.add_argument('--user-input-price-per-million', type=float)
    parser.add_argument('--user-output-price-per-million', type=float)
    args = parser.parse_args(argv)
    if min(args.n_tasks, args.max_steps, args.user_max_tokens) < 1:
        parser.error('Task count, step limit, and user token budget must be positive.')
    (seeds, models) = (args.seeds or [args.seed], args.models or [args.model])
    if len(set(seeds)) != len(seeds) or any((seed < 0 for seed in seeds)) or len(set(models)) != len(models) or any((not model for model in models)):
        parser.error('Seeds and models must be nonempty, unique, and valid.')
    user_prices = (args.user_input_price_per_million, args.user_output_price_per_million)
    if (user_prices[0] is None) != (user_prices[1] is None) or any((price is not None and (not math.isfinite(price) or price < 0) for price in user_prices)):
        parser.error('Supply both nonnegative finite simulator token prices or neither.')
    prototype = policy_from_args(args)
    (adapter, tasks) = modular_select_alfworld(args) if args.benchmark == 'alfworld' else modular_select_tau(args)
    validate_evaluation_ids(prototype.graph, [task['task_id'] for task in tasks])
    if prototype.graph is not None:
        benchmark = prototype.graph['provenance'].get('benchmark')
        if benchmark and benchmark != args.benchmark:
            parser.error('Graph provenance benchmark does not match the selected benchmark.')
    if args.benchmark != 'alfworld':
        importlib.import_module('litellm')
    versions = {}
    for package in ('alfworld', 'tau-bench', 'openai', 'litellm'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    out_dir = create_run_directory(args.out_dir)
    config = {key: str(value) if isinstance(value, Path) else value for (key, value) in vars(args).items()}
    manifest = {'benchmark': args.benchmark, 'protocol': 'supplied-interface modular LLM policy evaluation', 'config': config, 'models': models, 'seeds': seeds, 'tasks': tasks, 'versions': versions, 'graph_sha256': file_sha256(args.graph) if args.graph else None, 'graph_provenance': prototype.graph['provenance'] if prototype.graph else None, 'alfworld_config_sha256': file_sha256(args.alfworld_config) if args.alfworld_config else None, 'mechanism_backend': 'LLM predictions conditioned on supplied module descriptions and interface evidence', 'seed_scope': 'Environment RNG and requested model seeds; API determinism is provider-dependent.', 'simulator_model': args.user_model or args.model if args.benchmark != 'alfworld' else None, 'simulator_provider': args.user_provider if args.benchmark != 'alfworld' else None, 'cost_scope': 'Agent and benchmark-user usage are separate; costs require explicit token prices.'}
    write_json(out_dir / 'manifest.json', manifest)
    records = []
    for (model_index, model) in enumerate(models):
        for seed in seeds:
            current = argparse.Namespace(**vars(args))
            (current.model, current.seed) = (model, seed)
            current.user_model = args.user_model or args.model
            for task in tasks:
                policy = policy_from_args(current)
                record = modular_run_alfworld(current, adapter, task, policy) if args.benchmark == 'alfworld' else modular_run_tau(current, adapter, task, policy)
                record.update(benchmark=args.benchmark, model=model, seed=seed, condition=args.condition, attention_anchor=args.attention_anchor, immediate_feedback=args.immediate_feedback, control=policy.control, graph_sha256=manifest['graph_sha256'])
                filename = f"episode_m{model_index}_s{seed}_t{task['index']}.json"
                write_json(out_dir / filename, record)
                records.append({key: value for (key, value) in record.items() if key not in ('trajectory', 'model_calls', 'feedback')})
                write_json(out_dir / 'records.json', records)
                groups = [{'model': selected_model, 'seed': selected_seed, **modular_summarize([item for item in records if item['model'] == selected_model and item['seed'] == selected_seed])} for selected_model in models for selected_seed in seeds]
                write_json(out_dir / 'summary.json', {'overall': modular_summarize(records), 'groups': groups})
    print(json.dumps({'out_dir': str(out_dir), **modular_summarize(records)}, ensure_ascii=False, allow_nan=False))

SCIENCEWORLD_SOURCE_API = 'https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py'

def scienceworld_load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        tasks = json.load(handle)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Task manifest must be a nonempty JSON list.')
    identities = set()
    ids = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get('task_name'), str):
            raise ValueError('Each task needs task_name and variation_id.')
        variation = task.get('variation_id')
        if isinstance(variation, bool) or not isinstance(variation, int) or variation < 0:
            raise ValueError('variation_id must be a nonnegative integer.')
        identity = (task['task_name'], variation)
        task.setdefault('task_id', f"scienceworld:{task['task_name']}:{variation}")
        if not isinstance(task['task_id'], str) or not task['task_id'] or task['task_id'] in ids or (identity in identities):
            raise ValueError('Manifest contains duplicate or invalid task identifiers.')
        identities.add(identity)
        ids.add(task['task_id'])
    return tasks

def scienceworld_available_actions(env: Any, action_space: str) -> list[str]:
    if action_space == 'valid':
        actions = env.get_valid_action_object_combinations()
    else:
        (templates, _) = env.get_possible_action_object_combinations()
        actions = [item['action'] for item in templates]
    if not actions or not all((isinstance(item, str) and item for item in actions)):
        raise ValueError('ScienceWorld returned an empty or malformed action catalogue.')
    return sorted(set(actions))

def scienceworld_validate_tasks(env: Any, tasks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    names = set(env.get_task_names())
    for task in tasks:
        if task['task_name'] not in names:
            raise ValueError(f"Unknown ScienceWorld task: {task['task_name']}")
        env.load(task['task_name'], 0, args.simplifications, generateGoldPath=False)
        variations = set(getattr(env, f'get_variations_{args.split}')())
        if task['variation_id'] not in variations:
            raise ValueError(f"Task {task['task_id']} is not in the requested {args.split} split.")
        env.load(task['task_name'], task['variation_id'], args.simplifications, generateGoldPath=False)
        (observation, info) = env.reset()
        if not isinstance(observation, str) or not isinstance(info, dict) or 'score' not in info:
            raise ValueError('Unexpected ScienceWorld reset contract.')
        scienceworld_available_actions(env, args.action_space)

def scienceworld_evaluate_task(env: Any, task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    policy = policy_from_args(args)
    result = {'benchmark': 'scienceworld', 'task_id': task['task_id'], 'task_name': task['task_name'], 'variation_id': task['variation_id'], 'seed': args.seed, 'condition': args.condition, 'status': 'setup_error', 'success': None, 'score': None, 'reward': None, 'steps': 0, 'trajectory': [], 'errors': []}
    try:
        env.load(task['task_name'], task['variation_id'], args.simplifications, generateGoldPath=False)
        (observation, info) = env.reset()
        goal = env.get_task_description()
        score = float(info['score'])
    except Exception as error:
        result['errors'].append({'phase': 'environment_setup', 'type': type(error).__name__, 'message': str(error)})
        result.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(), model_calls=policy.calls)
        return result
    reward_sum = 0.0
    done = False
    result['status'] = 'running'
    phase = 'policy'
    try:
        for step in range(args.max_steps):
            phase = 'action_catalogue'
            actions = scienceworld_available_actions(env, args.action_space)
            phase = 'policy'
            decision = policy.choose_action(observation, goal, actions)
            phase = 'environment_step'
            (next_observation, reward, done, info) = env.step(decision['action'])
            score = float(info['score'])
            reward_sum += float(reward)
            result['steps'] = step + 1
            result['trajectory'].append({'step': step, 'observation': observation, 'decision': decision, 'next_observation': next_observation, 'reward': float(reward), 'score': score, 'done': bool(done), 'valid': info.get('valid')})
            observation = next_observation
            phase = 'feedback'
            policy.observe(decision['action'], next_observation, reward=float(reward), done=bool(done))
            if done:
                break
        result.update(status='completed', success=score >= 100.0, score=score, reward=reward_sum, termination='environment_done' if done else 'step_limit', environment_done=bool(done))
    except Exception as error:
        result.update(status='runtime_error', partial_score=score, partial_reward=reward_sum)
        result['errors'].append({'phase': phase, 'type': type(error).__name__, 'message': str(error)})
    result.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(), model_calls=policy.calls, feedback=policy.feedback)
    return result

def scienceworld_summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [record for record in records if record['status'] == 'completed']
    count = len(complete)
    usage = {key: sum((record['usage'][key] for record in records)) for key in ('calls', 'errors', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'elapsed_s', 'retries', 'provider_request_attempts', 'provider_successful_calls')}
    costs = [record['usage']['estimated_cost_usd'] for record in records]
    usage['estimated_cost_usd'] = sum(costs) if all((cost is not None for cost in costs)) else None
    return {'benchmark': 'scienceworld', 'selected_tasks': len(records), 'completed_tasks': count, 'setup_errors': sum((record['status'] == 'setup_error' for record in records)), 'runtime_errors': sum((record['status'] == 'runtime_error' for record in records)), 'mean_score_completed': sum((record['score'] for record in complete)) / count if count else None, 'success_rate_completed': sum((record['success'] for record in complete)) / count if count else None, 'success_denominator': count, 'all_selected_tasks_evaluated': count == len(records), 'mean_steps_completed': sum((record['steps'] for record in complete)) / count if count else None, 'mean_calls_completed': sum((record['usage']['calls'] for record in complete)) / count if count else None, 'mean_wall_time_s_completed': sum((record['wall_time_s'] for record in complete)) / count if count else None, 'usage_all_attempts': usage}

def scienceworld_evaluation_main(argv=None) -> None:
    _load_evaluation_helpers()
    parser = argparse.ArgumentParser()
    add_policy_arguments(parser)
    parser.add_argument('--task-manifest', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'dev', 'test'), default='test')
    parser.add_argument('--jar-path', type=Path)
    parser.add_argument('--simplifications', default='')
    parser.add_argument('--action-space', choices=('possible', 'valid'), default='possible')
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--outdir', '--out-dir', dest='outdir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error('--max-steps must be positive.')
    tasks = scienceworld_load_tasks(args.task_manifest)
    prototype = policy_from_args(args)
    validate_evaluation_ids(prototype.graph, [task['task_id'] for task in tasks])
    if prototype.graph is not None:
        graph_benchmark = prototype.graph['provenance'].get('benchmark')
        if graph_benchmark is not None and str(graph_benchmark).lower().replace('-', '').replace('_', '') != 'scienceworld':
            raise ValueError('Graph benchmark provenance does not match ScienceWorld.')
    run_dir = create_run_directory(args.outdir)
    metadata = {'arguments': {key: str(value) if isinstance(value, Path) else value for (key, value) in vars(args).items()}, 'task_manifest_sha256': file_sha256(args.task_manifest), 'tasks': tasks, 'source_api': SCIENCEWORLD_SOURCE_API, 'graph_sha256': file_sha256(args.graph) if args.graph else None, 'protocol': 'ScienceWorld text interaction; real simulator score, no gold path or hidden object tree'}
    write_json(run_dir / 'configuration.json', metadata)
    env = None
    records = []
    try:
        from scienceworld import ScienceWorldEnv
        metadata['scienceworld_version'] = importlib.metadata.version('scienceworld')
        from scienceworld.constants import JAR_PATH
        metadata['simulator_jar_sha256'] = file_sha256(args.jar_path or JAR_PATH)
        env = ScienceWorldEnv('', str(args.jar_path) if args.jar_path else None, envStepLimit=args.max_steps + 1)
        scienceworld_validate_tasks(env, tasks, args)
        write_json(run_dir / 'configuration.json', metadata)
    except Exception as error:
        write_json(run_dir / 'summary.json', {'status': 'setup_error', 'selected_tasks': len(tasks), 'completed_tasks': 0, 'error_type': type(error).__name__, 'error': str(error)})
        if env is not None:
            env.close()
        raise
    try:
        with (run_dir / 'episodes.jsonl').open('x', encoding='utf-8') as output:
            for task in tasks:
                record = scienceworld_evaluate_task(env, task, args)
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                output.flush()
                write_json(run_dir / 'summary.json', scienceworld_summarize(records))
        write_json(run_dir / 'summary.json', scienceworld_summarize(records))
    finally:
        env.close()

APIBANK_SOURCE_API = 'https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank'
APIBANK_PROTOCOL = 'stateful_dialogue_fixed_call_slots'

def apibank_load_dialogues(data_dir: Path, manifest_path: Path | None) -> list[dict[str, Any]]:
    data_dir = data_dir.resolve(strict=True)
    if manifest_path is None:
        paths = sorted(data_dir.rglob('*.jsonl'))
    else:
        with manifest_path.open(encoding='utf-8') as handle:
            relative_paths = json.load(handle)
        if not isinstance(relative_paths, list) or not relative_paths or (not all((isinstance(item, str) for item in relative_paths))):
            raise ValueError('Dialogue manifest must be a nonempty JSON list of relative JSONL paths.')
        paths = [(data_dir / item).resolve(strict=True) for item in relative_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('No dialogue files selected, or duplicate files selected.')
    tasks = []
    for path in paths:
        if not path.is_relative_to(data_dir) or path.suffix != '.jsonl':
            raise ValueError('Dialogue paths must be JSONL files inside --data-dir.')
        with path.open(encoding='utf-8') as handle:
            turns = [json.loads(line) for line in handle if line.strip()]
        has_user = False
        calls = 0
        for turn in turns:
            if not isinstance(turn, dict) or turn.get('role') not in ('User', 'AI', 'API'):
                raise ValueError(f'Malformed dialogue turn in {path}.')
            if turn['role'] in ('User', 'AI'):
                if not isinstance(turn.get('text'), str):
                    raise ValueError(f'Text is required for dialogue turns in {path}.')
                has_user |= turn['role'] == 'User'
            else:
                if not has_user or not isinstance(turn.get('api_name'), str) or (not isinstance(turn.get('param_dict'), dict)):
                    raise ValueError(f'API turn lacks an observed user goal or valid API fields in {path}.')
                result = turn.get('result')
                if not isinstance(result, dict) or 'output' not in result or 'exception' not in result:
                    raise ValueError(f'API turn lacks an official reference result in {path}.')
                calls += 1
        if not calls:
            raise ValueError(f'No API calls in selected dialogue: {path}')
        relative_path = path.relative_to(data_dir).as_posix()
        tasks.append({'task_id': 'apibank:' + relative_path, 'relative_path': relative_path, 'source_sha256': file_sha256(path), 'turns': turns, 'total_calls': calls})
    return tasks

def apibank_load_tool_manager(benchmark_dir: Path) -> Any:
    expected = (benchmark_dir / 'tool_manager.py').resolve(strict=True)
    if not (benchmark_dir / 'apis').is_dir() or not (benchmark_dir / 'init_database').is_dir():
        raise ValueError('--benchmark-dir must point to the upstream api-bank directory.')
    sys.path.insert(0, str(benchmark_dir))
    if 'apis' in sys.modules:
        origin = Path(sys.modules['apis'].__file__).resolve()
        if not origin.is_relative_to(benchmark_dir):
            raise RuntimeError('Another package named apis is already imported; run this evaluator in a fresh process.')
    specification = importlib.util.spec_from_file_location('fed_causal_apibank_tool_manager', expected)
    if specification is None or specification.loader is None:
        raise ImportError('Cannot load the upstream APIBank ToolManager.')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.ToolManager

def apibank_required_parameters(manager: Any, name: str) -> list[str]:
    tool_class = manager.get_api_by_name(name)['class']
    signature = inspect.signature(tool_class.call)
    return [parameter.name for parameter in signature.parameters.values() if parameter.name != 'self' and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY) and (parameter.default is inspect.Parameter.empty)]

def apibank_tool_catalogue(manager: Any) -> list[dict[str, Any]]:
    catalogue = []
    type_map = {'str': 'string', 'int': 'integer', 'float': 'number', 'bool': 'boolean', 'list': 'array', 'list(str)': 'array'}
    for name in sorted(manager.list_all_apis()):
        description = json.loads(manager.get_api_description(name))
        properties = {}
        for (key, parameter) in description['input_parameters'].items():
            item = {'description': parameter.get('description', '')}
            if parameter['type'] in type_map:
                item['type'] = type_map[parameter['type']]
            else:
                item['description'] += f" Declared upstream type: {parameter['type']}."
            if parameter['type'] == 'list(str)':
                item['items'] = {'type': 'string'}
            properties[key] = item
        catalogue.append({'type': 'function', 'function': {'name': name, 'description': description['description'], 'parameters': {'type': 'object', 'properties': properties, 'required': apibank_required_parameters(manager, name), 'additionalProperties': False}}})
    if not catalogue:
        raise ValueError('The upstream API catalogue is empty.')
    return catalogue

def apibank_validate_tasks(manager: Any, tasks: list[dict[str, Any]]) -> None:
    names = set(manager.list_all_apis())
    for task in tasks:
        for turn in task['turns']:
            if turn['role'] != 'API':
                continue
            if turn['api_name'] not in names:
                raise ValueError(f"Unknown reference tool {turn['api_name']} in {task['task_id']}")
            parameter_names = set(manager.get_api_by_name(turn['api_name'])['input_parameters'])
            provided_parameters = set(turn['param_dict'])
            required = set(apibank_required_parameters(manager, turn['api_name']))
            if not provided_parameters.issubset(parameter_names) or not required.issubset(provided_parameters):
                raise ValueError(f"Reference parameter schema mismatch in {task['task_id']}:{turn['api_name']}")
            tool = manager.init_tool(turn['api_name'])
            if not callable(getattr(tool, 'check_api_call_correctness', None)):
                raise ValueError(f"Upstream correctness checker missing for {turn['api_name']}")

def apibank_stochastic_reference_tools(manager: Any) -> dict[str, list[str]]:
    result = {}
    ignored_random_calls = {'seed', 'getstate', 'setstate'}
    for name in sorted(manager.list_all_apis()):
        tool_class = manager.get_api_by_name(name)['class']
        module_path = inspect.getsourcefile(tool_class)
        if module_path is None:
            raise ValueError(f'Cannot inspect reference compatibility for upstream API {name}.')
        module_tree = ast.parse(Path(module_path).read_text(encoding='utf-8'))
        aliases = {}
        direct = {}
        for node in ast.walk(module_tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name in ('random', 'uuid', 'secrets'):
                        aliases[imported.asname or imported.name] = imported.name
            elif isinstance(node, ast.ImportFrom) and node.module in ('random', 'uuid', 'secrets'):
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
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and (target.value.id in aliases):
                origin = (aliases[target.value.id], target.attr)
            if origin is not None and (not (origin[0] == 'random' and origin[1] in ignored_random_calls)):
                generators.add('.'.join(origin))
        if generators:
            result[name] = sorted(generators)
    return result

def apibank_execute_tool(manager: Any, action: dict[str, Any]) -> Any:
    parameters = dict(action['arguments'])
    specification = manager.get_api_by_name(action['name'])['input_parameters']
    for (name, value) in parameters.items():
        if specification[name]['type'] == 'bool' and isinstance(value, bool):
            parameters[name] = 'True' if value else 'False'
    return manager.api_call(action['name'], **parameters)

def apibank_evaluate_dialogue(manager_type: Any, catalogue: list[dict[str, Any]], task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    policy = policy_from_args(args)
    record = {'benchmark': 'apibank', 'protocol': APIBANK_PROTOCOL, 'task_id': task['task_id'], 'source_sha256': task['source_sha256'], 'seed': args.seed, 'condition': args.condition, 'status': 'setup_error', 'success': None, 'reward': None, 'call_accuracy': None, 'correct_calls': 0, 'total_calls': task['total_calls'], 'steps': 0, 'trajectory': [], 'errors': []}
    incompatible = task.get('incompatible_reference_apis', {})
    if incompatible:
        record['status'] = 'protocol_incompatible'
        record['errors'].append({'phase': 'protocol_validation', 'type': 'StochasticReferenceMismatch', 'message': 'Upstream APIs generate stochastic values, so saved-reference exact result checks are not validated for persistent regenerated state. No calls were executed and no gold identifiers were injected.', 'apis': incompatible})
        record.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(), model_calls=[], feedback=[])
        return record
    phase = 'environment_setup'
    environment_seed = (args.seed + int(hashlib.sha256(task['task_id'].encode()).hexdigest()[:16], 16)) % 2 ** 32
    record['environment_seed'] = environment_seed
    previous_random_state = random.getstate()
    random.seed(environment_seed)
    try:
        manager = manager_type()
        history = []
        user_turns = []
        record['status'] = 'running'
        for turn in task['turns']:
            if turn['role'] == 'AI':
                continue
            if turn['role'] == 'User':
                user_turns.append(turn['text'])
                history.append({'role': 'user', 'content': turn['text']})
                continue
            phase = 'policy'
            goal = "Complete the user's currently observed requests through the available API tools. The benchmark reference year is 2023.\n" + '\n'.join(user_turns)
            observation = {'observed_conversation': history}
            decision = policy.choose_action(observation, goal, catalogue, history=history)
            action = decision['action']
            phase = 'tool_execution'
            result = apibank_execute_tool(manager, action)
            record['steps'] += 1
            history.append({'role': 'assistant', 'tool_call': action})
            history.append({'role': 'tool', 'name': action['name'], 'content': result})
            entry = {'call_index': record['steps'] - 1, 'decision': decision, 'tool_result': result, 'correct': None, 'reference_api': turn['api_name']}
            record['trajectory'].append(entry)
            phase = 'official_scoring'
            if action['name'] == turn['api_name']:
                checker = manager.init_tool(action['name'])
                correct = checker.check_api_call_correctness(copy.deepcopy(result), copy.deepcopy(turn['result']))
                if not isinstance(correct, bool):
                    raise ValueError('Official APIBank checker did not return a boolean.')
            else:
                correct = False
            entry['correct'] = correct
            record['correct_calls'] += int(correct)
            phase = 'feedback'
            policy.observe(action, result, reward=None, done=record['steps'] == task['total_calls'])
        record.update(status='completed', success=record['correct_calls'] == task['total_calls'], call_accuracy=record['correct_calls'] / task['total_calls'], reward=record['correct_calls'] / task['total_calls'])
    except Exception as error:
        record['status'] = 'setup_error' if phase == 'environment_setup' else 'runtime_error'
        record['errors'].append({'phase': phase, 'type': type(error).__name__, 'message': str(error)})
    random.setstate(previous_random_state)
    record.update(wall_time_s=time.perf_counter() - started, usage=policy.snapshot_usage(), model_calls=policy.calls, feedback=policy.feedback)
    return record

def apibank_summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [record for record in records if record['status'] == 'completed']
    count = len(complete)
    denominator = sum((record['total_calls'] for record in complete))
    usage = {key: sum((record['usage'][key] for record in records)) for key in ('calls', 'errors', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'elapsed_s', 'retries', 'provider_request_attempts', 'provider_successful_calls')}
    costs = [record['usage']['estimated_cost_usd'] for record in records]
    usage['estimated_cost_usd'] = sum(costs) if all((cost is not None for cost in costs)) else None
    return {'benchmark': 'apibank', 'protocol': APIBANK_PROTOCOL, 'selected_dialogues': len(records), 'completed_dialogues': count, 'setup_errors': sum((record['status'] == 'setup_error' for record in records)), 'runtime_errors': sum((record['status'] == 'runtime_error' for record in records)), 'protocol_incompatible_dialogues': sum((record['status'] == 'protocol_incompatible' for record in records)), 'all_calls_correct_rate_completed': sum((record['success'] for record in complete)) / count if count else None, 'micro_call_accuracy_completed': sum((record['correct_calls'] for record in complete)) / denominator if denominator else None, 'macro_call_accuracy_completed': sum((record['call_accuracy'] for record in complete)) / count if count else None, 'call_accuracy_denominator': denominator, 'dialogue_success_denominator': count, 'all_selected_dialogues_evaluated': count == len(records), 'mean_calls_completed': sum((record['usage']['calls'] for record in complete)) / count if count else None, 'mean_wall_time_s_completed': sum((record['wall_time_s'] for record in complete)) / count if count else None, 'usage_all_attempts': usage}

def apibank_evaluation_main(argv=None) -> None:
    _load_evaluation_helpers()
    parser = argparse.ArgumentParser()
    add_policy_arguments(parser)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--dialogue-manifest', type=Path)
    parser.add_argument('--outdir', '--out-dir', dest='outdir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.benchmark_dir = args.benchmark_dir.expanduser().resolve(strict=True)
    args.data_dir = args.data_dir.expanduser().resolve(strict=True)
    args.outdir = args.outdir.expanduser().resolve()
    if args.graph:
        args.graph = args.graph.expanduser().resolve(strict=True)
    if args.dialogue_manifest:
        args.dialogue_manifest = args.dialogue_manifest.expanduser().resolve(strict=True)
    tasks = apibank_load_dialogues(args.data_dir, args.dialogue_manifest)
    prototype = policy_from_args(args)
    validate_evaluation_ids(prototype.graph, [task['task_id'] for task in tasks])
    if prototype.graph is not None:
        graph_benchmark = prototype.graph['provenance'].get('benchmark')
        if graph_benchmark is not None and str(graph_benchmark).lower().replace('-', '').replace('_', '') not in ('apibank', 'apibankstateful'):
            raise ValueError('Graph benchmark provenance does not match APIBank.')
    run_dir = create_run_directory(args.outdir)
    selected = [{key: value for (key, value) in task.items() if key != 'turns'} for task in tasks]
    digest_payload = {task['relative_path']: task['source_sha256'] for task in tasks}
    metadata = {'benchmark': 'apibank', 'protocol': APIBANK_PROTOCOL, 'source_api': APIBANK_SOURCE_API, 'protocol_description': 'Persistent upstream ToolManager state per dialogue; replay only observed user turns; choose one tool at each reference API slot; score with upstream API correctness checkers. Reference AI text, target tools, parameters, and results are never policy inputs. This is a fixed-call-slot diagnostic, not a free-dialogue end-to-end benchmark.', 'arguments': {key: str(value) if isinstance(value, Path) else value for (key, value) in vars(args).items()}, 'tasks': selected, 'dataset_sha256': hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest(), 'graph_sha256': file_sha256(args.graph) if args.graph else None, 'environment_seed_scope': 'Python random reset per dialogue to (seed + SHA256(task_id) first 64 bits) modulo 2^32; separate from policy request seed', 'parameter_schema_policy': 'Required parameters derived from upstream call signatures; unsupported JSON-schema type names preserve upstream descriptions and are checked by ToolManager at execution'}
    write_json(run_dir / 'configuration.json', metadata)
    previous_directory = Path.cwd()
    records = []
    try:
        os.chdir(args.benchmark_dir)
        try:
            manager_type = apibank_load_tool_manager(args.benchmark_dir)
            manager = manager_type()
            catalogue = apibank_tool_catalogue(manager)
            apibank_validate_tasks(manager, tasks)
            stochastic = apibank_stochastic_reference_tools(manager)
            for task in tasks:
                task['incompatible_reference_apis'] = {turn['api_name']: stochastic[turn['api_name']] for turn in task['turns'] if turn['role'] == 'API' and turn['api_name'] in stochastic}
            metadata['tasks'] = [{key: value for (key, value) in task.items() if key != 'turns'} for task in tasks]
            metadata['reference_compatibility_policy'] = 'Conservatively exclude dialogues whose APIs contain executable random/UUID/secrets generation; report protocol_incompatible without injecting saved identifiers or changing upstream checkers.'
            metadata['tool_manager_sha256'] = file_sha256(args.benchmark_dir / 'tool_manager.py')
            metadata['api_source_sha256'] = {path.name: file_sha256(path) for path in sorted((args.benchmark_dir / 'apis').glob('*.py'))}
            metadata['initial_database_sha256'] = {path.name: file_sha256(path) for path in sorted((args.benchmark_dir / 'init_database').glob('*.json'))}
            metadata['tool_catalogue'] = catalogue
            write_json(run_dir / 'configuration.json', metadata)
        except Exception as error:
            write_json(run_dir / 'summary.json', {'status': 'setup_error', 'selected_dialogues': len(tasks), 'completed_dialogues': 0, 'error_type': type(error).__name__, 'error': str(error)})
            raise
        with (run_dir / 'episodes.jsonl').open('x', encoding='utf-8') as output:
            for task in tasks:
                record = apibank_evaluate_dialogue(manager_type, catalogue, task, args)
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                output.flush()
                write_json(run_dir / 'summary.json', apibank_summarize(records))
        write_json(run_dir / 'summary.json', apibank_summarize(records))
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    evaluation_modes = {
        "--modular-evaluation": modular_evaluation_main,
        "--scienceworld-evaluation": scienceworld_evaluation_main,
        "--apibank-evaluation": apibank_evaluation_main,
    }
    selected_modes = [flag for flag in evaluation_modes if flag in sys.argv[1:]]
    if len(selected_modes) > 1:
        raise SystemExit("Select exactly one evaluation mode.")
    if selected_modes:
        flag = selected_modes[0]
        sys.exit(evaluation_modes[flag]([argument for argument in sys.argv[1:] if argument != flag]))
    sys.exit(main())
