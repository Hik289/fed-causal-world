from __future__ import annotations

import os
import threading
import time
from typing import Any
import argparse
import hashlib
import json
import math
from pathlib import Path

API_KEY = os.environ.get("FED_CAUSAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
API_BASE_URL = os.environ.get("FED_CAUSAL_API_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL"
)
DEPLOYMENT_NAME = os.environ.get("FED_CAUSAL_MODEL", "gpt-5.4-mini")

PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00


_client = None
_lock = threading.Lock()
_counter = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "usd_estimated": 0.0,
    "errors": 0,
    "retries": 0,
}


def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "Set FED_CAUSAL_API_KEY or OPENAI_API_KEY before API-backed runs."
            )
        from openai import OpenAI

        kwargs = {"api_key": API_KEY}
        if API_BASE_URL:
            kwargs["base_url"] = API_BASE_URL
        _client = OpenAI(**kwargs)
    return _client


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_INPUT_PER_1M
            + completion_tokens * PRICE_OUTPUT_PER_1M) / 1_000_000.0


def chat(messages: list[dict[str, str]],
         max_tokens: int = 512,
         temperature: float = 0.0,
         model: str = DEPLOYMENT_NAME,
         max_retries: int = 4,
         timeout: float = 60.0,
         **kwargs) -> tuple[str, dict[str, Any]]:
    client = get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    **kwargs,
                )
            except TypeError:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    **kwargs,
                )
            elapsed = time.time() - t0

            text = resp.choices[0].message.content or ""
            u = resp.usage
            pt = getattr(u, "prompt_tokens", 0) or 0
            ct = getattr(u, "completion_tokens", 0) or 0
            tt = getattr(u, "total_tokens", pt + ct) or (pt + ct)
            usd = _estimate_cost(pt, ct)
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "usd_estimated": usd,
                "elapsed_s": elapsed,
            }
            with _lock:
                _counter["calls"] += 1
                _counter["prompt_tokens"] += pt
                _counter["completion_tokens"] += ct
                _counter["total_tokens"] += tt
                _counter["usd_estimated"] += usd
            return text, usage
        except Exception as e:
            last_err = e
            msg = str(e)
            with _lock:
                _counter["retries"] += 1
            if "429" in msg or "rate" in msg.lower():
                time.sleep(min(2 ** attempt, 10))
            else:
                time.sleep(0.5 * (attempt + 1))
            continue
    with _lock:
        _counter["errors"] += 1
    raise RuntimeError(f"LLM chat failed after {max_retries} retries: {last_err}")


def reset_counter():
    with _lock:
        for k in _counter:
            _counter[k] = 0 if isinstance(_counter[k], int) else 0.0


def get_counter() -> dict[str, Any]:
    with _lock:
        return dict(_counter)


CONDITIONS = ("sequence", "discovered", "oracle", "causalcompose")


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def create_run_directory(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_graph(payload: Any, condition: str) -> dict[str, Any] | None:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "sequence":
        if payload is not None:
            raise ValueError("The sequence condition must not receive a graph.")
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{condition} requires an explicit graph JSON object.")
    modules = payload.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Graph modules must be a nonempty mapping.")
    for name, specification in modules.items():
        if not isinstance(name, str) or not name or not isinstance(specification, (str, dict)):
            raise ValueError("Each module needs a name and a text or object specification.")
    interfaces = payload.get("interfaces")
    if not isinstance(interfaces, list):
        raise ValueError("Graph interfaces must be a list.")
    seen = set()
    for edge in interfaces:
        if not isinstance(edge, dict):
            raise ValueError("Each interface must be an object.")
        for key in ("source", "target"):
            if edge.get(key) not in modules:
                raise ValueError(f"Unknown interface {key}: {edge.get(key)}")
        for key in ("source_event", "target_event"):
            if not isinstance(edge.get(key), str) or not edge[key]:
                raise ValueError(f"An interface requires {key}.")
        for key in ("lag_min", "lag_max", "support", "opportunities"):
            value = edge.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Interface {key} must be finite and nonnegative.")
        if edge["lag_max"] < edge["lag_min"] or edge["support"] > edge["opportunities"]:
            raise ValueError("Invalid interface lag range or support count.")
        identity = tuple(edge[key] for key in ("source", "target", "source_event", "target_event"))
        if identity in seen:
            raise ValueError(f"Duplicate interface: {identity}")
        seen.add(identity)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Graph provenance must describe its source.")
    kind = provenance.get("kind")
    if condition == "oracle":
        if kind not in ("oracle", "benchmark_oracle", "manually_specified_oracle"):
            raise ValueError("Oracle condition requires explicit oracle provenance.")
        if not provenance.get("source"):
            raise ValueError("Oracle provenance requires a nonempty source.")
    elif kind != "intervention_response_discovery":
        raise ValueError("Discovered conditions require intervention_response_discovery provenance.")
    return json.loads(json.dumps(payload))


def load_graph(path: str | Path | None, condition: str) -> dict[str, Any] | None:
    if condition == "sequence":
        return validate_graph(None if path is None else {}, condition)
    if path is None:
        raise ValueError(f"--graph is required for {condition}.")
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        return validate_graph(json.load(handle), condition)


def validate_evaluation_ids(graph: dict[str, Any] | None, task_ids: list[str]) -> None:
    if graph is None:
        return
    provenance = graph["provenance"]
    if provenance["kind"] != "intervention_response_discovery":
        return
    training_ids = provenance.get("training_task_ids")
    if not isinstance(training_ids, list) or not training_ids:
        raise ValueError("Discovered benchmark graphs require provenance.training_task_ids to audit split separation.")
    if not all(isinstance(item, str) and item for item in training_ids):
        raise ValueError("Graph training task IDs must be nonempty strings.")
    overlap = set(map(str, task_ids)).intersection(training_ids)
    if overlap:
        raise ValueError(f"Graph training/evaluation task overlap: {sorted(overlap)}")
    inputs = provenance.get("input_sha256")
    if not isinstance(inputs, dict) or not inputs or not all(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower()) for value in inputs.values()):
        raise ValueError("Graph provenance requires source input SHA256 hashes.")
    if str(provenance.get("task_split", "")).lower() not in ("train", "training"):
        raise ValueError("Graph discovery requires explicit train/training split provenance.")


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--model", default=os.environ.get("FED_CAUSAL_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-anchor", action="store_true")
    parser.add_argument("--immediate-feedback", action="store_true")
    parser.add_argument("--no-control", dest="control", action="store_false", default=True)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-rollout-modules", type=int, default=12)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)


def policy_from_args(args: argparse.Namespace) -> "AgentPolicy":
    prices = (args.input_price_per_million, args.output_price_per_million)
    if (prices[0] is None) != (prices[1] is None):
        raise ValueError("Supply both token prices or neither.")
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in prices):
        raise ValueError("Token prices must be finite and nonnegative.")
    return AgentPolicy(load_graph(args.graph, args.condition), args.condition, args.model, args.seed,
                       attention_anchor=args.attention_anchor, immediate_feedback=args.immediate_feedback,
                       control=args.control, max_tokens=args.max_tokens,
                       max_rollout_modules=args.max_rollout_modules,
                       input_price_per_million=prices[0], output_price_per_million=prices[1])


class AgentPolicy:
    def __init__(self, graph_payload: dict[str, Any] | None, condition: str, model: str, seed: int,
                 attention_anchor: bool = False, immediate_feedback: bool = False, control: bool = True,
                 max_tokens: int = 1200, max_rollout_modules: int = 12,
                 input_price_per_million: float | None = None, output_price_per_million: float | None = None):
        self.graph = validate_graph(graph_payload, condition)
        if not model or max_tokens < 1 or max_rollout_modules < 1:
            raise ValueError("Model, max_tokens and max_rollout_modules must be valid.")
        if (input_price_per_million is None) != (output_price_per_million is None):
            raise ValueError("Both token prices are required for cost estimation.")
        for price in (input_price_per_million, output_price_per_million):
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ValueError("Token prices must be finite and nonnegative.")
        self.condition = condition
        self.model = model
        self.seed = seed
        self.attention_anchor = attention_anchor
        self.immediate_feedback = immediate_feedback
        self.control = bool(control and condition == "causalcompose")
        self.max_tokens = max_tokens
        self.max_rollout_modules = max_rollout_modules
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million
        self.calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.last_decision: dict[str, Any] | None = None
        self.feedback: list[dict[str, Any]] = []

    def _query(self, phase: str, instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        record = {"phase": phase, "status": "started", "prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0, "seed": self.seed + len(self.calls)}
        self.calls.append(record)
        counters_before = get_counter()
        try:
            text, usage = chat(
                [{"role": "system", "content": instruction + " Return one JSON object, without Markdown."},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}],
                max_tokens=self.max_tokens, model=self.model, temperature=0.0, seed=record["seed"])
            record.update({key: usage.get(key, 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens")})
            record["response"] = text
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("Model response must be a JSON object.")
            record["status"] = "ok"
            return result
        except Exception as error:
            record.update(status="error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record["elapsed_s"] = time.perf_counter() - started
            counters_after = get_counter()
            record["retries"] = counters_after["retries"] - counters_before["retries"]
            record["provider_errors"] = counters_after["errors"] - counters_before["errors"]
            record["provider_successful_calls"] = counters_after["calls"] - counters_before["calls"]
            record["provider_request_attempts"] = record["provider_successful_calls"] + record["retries"]

    @staticmethod
    def _validate_action(action: Any, available_actions: list[Any]) -> Any:
        if not available_actions:
            raise ValueError("The environment supplied no actions.")
        if all(isinstance(item, str) for item in available_actions):
            if not isinstance(action, str) or action not in available_actions:
                raise ValueError(f"Action is not an admissible command: {action!r}")
            return action
        if not isinstance(action, dict) or not isinstance(action.get("name"), str) or not isinstance(action.get("arguments"), dict):
            raise ValueError("Tool actions must be objects with name and arguments.")
        specifications = {}
        for item in available_actions:
            if not isinstance(item, dict):
                raise ValueError("Mixed command/tool action catalogues are unsupported.")
            function = item.get("function", item)
            specifications[function["name"]] = function
        if action["name"] not in specifications:
            raise ValueError(f"Unknown tool: {action['name']}")
        schema = specifications[action["name"]].get("parameters", {})
        arguments = action["arguments"]
        missing = set(schema.get("required", [])).difference(arguments)
        if missing:
            raise ValueError(f"Missing tool arguments: {sorted(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(arguments).difference(properties):
            raise ValueError("Tool action includes unsupported arguments.")
        for key, value in arguments.items():
            specification = properties.get(key, {})
            expected = specification.get("type")
            types = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
            if expected in types and (not isinstance(value, types[expected]) or expected in ("integer", "number") and isinstance(value, bool)):
                raise ValueError(f"Incorrect type for tool argument {key}.")
            if "enum" in specification and value not in specification["enum"]:
                raise ValueError(f"Unsupported value for tool argument {key}.")
        return {"name": action["name"], "arguments": arguments}

    def _compose(self, proposal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        module = proposal.get("module")
        if module not in self.graph["modules"]:
            raise ValueError("Causal composition requires a valid source module in the proposal.")
        queue = [{"module": module, "time": 0.0, "incoming": None}]
        predictions = {}
        updates = []
        routed_edges = []
        while queue and len(updates) < self.max_rollout_modules:
            queue.sort(key=lambda item: (item["time"], item["module"]))
            first = queue.pop(0)
            current, event_time = first["module"], first["time"]
            batch = [first]
            remaining = []
            for item in queue:
                if item["module"] == current and item["time"] == event_time:
                    batch.append(item)
                else:
                    remaining.append(item)
            queue = remaining
            incoming = [item["incoming"] for item in batch if item["incoming"] is not None]
            is_initial_intervention = not updates
            result = self._query("local_mechanism_prediction",
                                 "Predict this module's response from its local mechanism description, current evidence, previous predicted local state, and newly routed upstream events. "
                                 "Do not treat predicted events as observations. Identify unmet prerequisites and uncertainty. "
                                 "Emit only newly triggered events, not unchanged state properties. "
                                 "Return prediction, emitted_events (list of event names), prerequisites_met (true, false, or null), and evidence.",
                                 {"module": current, "mechanism": self.graph["modules"][current], "incoming": incoming,
                                  "previous_local_prediction": predictions.get(current), "scenario_time": event_time,
                                  "candidate_action": proposal["action"] if is_initial_intervention else None,
                                  "outgoing_event_names": sorted({edge["source_event"] for edge in self.graph["interfaces"] if edge["source"] == current}),
                                  "observation": context["observation"], "observed_history": context["history"]})
            if not isinstance(result.get("emitted_events"), list) or not all(isinstance(x, str) for x in result["emitted_events"]):
                raise ValueError("Local prediction must include a list of emitted_events.")
            if result.get("prerequisites_met") is not None and not isinstance(result["prerequisites_met"], bool):
                raise ValueError("Invalid predicted prerequisite status.")
            predictions[current] = result
            updates.append({"module": current, "time": event_time, "incoming": incoming, "prediction": result})
            for edge in self.graph["interfaces"]:
                if edge["source"] == current and edge["source_event"] in result["emitted_events"]:
                    arrival = event_time + edge["lag_min"]
                    routed_edges.append({"interface": edge, "source_time": event_time, "arrival_time": arrival})
                    queue.append({"module": edge["target"], "time": arrival,
                                  "incoming": {"interface": edge, "source_prediction": result,
                                               "source_time": event_time, "source_update": len(updates) - 1}})
        return {"module_predictions": predictions, "prediction_updates": updates, "routed_interfaces": routed_edges,
                "unexpanded_events": queue, "truncated": bool(queue),
                "timing_policy": "earliest-arrival scenario at lag_min; lag_max is retained as timing uncertainty",
                "cycle_policy": "event-driven re-entry bounded by max_rollout_modules prediction updates",
                "prediction_source": "LLM local mechanism hypotheses routed over supplied interfaces"}

    def choose_action(self, observation: Any, goal: str, available_actions: list[Any],
                      history: list[Any] | None = None) -> dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("A nonempty task goal is required.")
        context = {"observation": observation, "goal": goal, "available_actions": available_actions,
                   "history": history if history is not None else self.events,
                   "feedback": self.feedback if self.immediate_feedback else []}
        if self.attention_anchor:
            context["decision_anchor"] = {"current_goal": goal,
                "instruction": "Use only currently observed prerequisites and the dependencies relevant to this goal when choosing the next action."}
        if self.graph is not None:
            context["world_model"] = {"modules": self.graph["modules"], "interfaces": self.graph["interfaces"],
                                      "provenance": self.graph["provenance"]}
        description = {
            "sequence": "Predict from the observed event sequence. No causal graph has been supplied.",
            "discovered": "Use the supplied graph estimated from intervention-response training traces; its edges can be uncertain. Graph guidance is prediction-only.",
            "oracle": "Use the supplied benchmark- or manually specified oracle graph. Its provenance is supplied; do not infer oracle status from predictions.",
            "causalcompose": "Propose a local action and identify its source module. Downstream module predictions will be computed separately."}[self.condition]
        proposal = self._query("propose", description + " Choose one admissible action. "
                               "Return action, module (the module name, or null without a graph), prediction, and a brief reason. "
                               "For command lists action must be an exact string; for tool schemas action must have name and arguments.", context)
        proposal["action"] = self._validate_action(proposal.get("action"), available_actions)
        diagnostics = {"condition": self.condition, "proposal": proposal, "control_enabled": self.control,
                       "attention_anchor": self.attention_anchor, "immediate_feedback": self.immediate_feedback}
        action = proposal["action"]
        if self.condition == "causalcompose":
            composition = self._compose(proposal, context)
            diagnostics["composition"] = composition
            if self.control:
                gate = self._query("check_and_gate",
                                   "Check the candidate against observed prerequisites and the composed predictions. "
                                   "Predictions are hypotheses, not ground truth. Keep the candidate when supported; otherwise choose an admissible prerequisite or information-gathering action. "
                                   "Return action, candidate_accepted (boolean), and reason.",
                                   {**context, "candidate": proposal, "composition": composition})
                action = self._validate_action(gate.get("action"), available_actions)
                if not isinstance(gate.get("candidate_accepted"), bool):
                    raise ValueError("Controller must report candidate_accepted as a boolean.")
                if gate["candidate_accepted"] != (action == proposal["action"]):
                    raise ValueError("Controller acceptance flag conflicts with its selected action.")
                diagnostics["gate"] = gate
        decision = {"action": action, "diagnostics": diagnostics}
        self.last_decision = decision
        return decision

    def observe(self, action: Any, observation: Any, reward: float | None = None, done: bool = False) -> None:
        event = {"action": action, "observation": observation, "reward": reward, "done": done}
        self.events.append(event)
        if self.immediate_feedback and self.last_decision is not None and not done:
            decision = self.last_decision
            diagnostics = decision["diagnostics"]
            prediction_applies = action == diagnostics["proposal"]["action"]
            feedback = self._query("verify_feedback",
                                   "Compare the actual observed outcome with the earlier prediction when it concerns the executed action. "
                                   "If the controller revised the action, do not score the unexecuted candidate's prediction. "
                                   "Return verified_observations, prediction_errors, and next_action_constraints. Do not invent hidden state.",
                                   {"executed": event, "prediction_applies_to_executed_action": prediction_applies,
                                    "earlier_prediction": diagnostics if prediction_applies else None})
            self.feedback.append(feedback)

    def snapshot_usage(self) -> dict[str, Any]:
        usage = {key: sum(call.get(key, 0) for call in self.calls)
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens", "elapsed_s", "retries", "provider_errors", "provider_successful_calls", "provider_request_attempts")}
        usage.update(calls=len(self.calls), errors=sum(call["status"] == "error" for call in self.calls),
                     successful_calls=sum(call["status"] == "ok" for call in self.calls))
        usage["calls_unit"] = "logical policy queries; provider_request_attempts includes failed/retried requests"
        usage["token_accounting"] = "reported successful-response usage; failed-request token usage unavailable"
        usage["estimated_cost_usd"] = None if self.input_price is None else (
            usage["prompt_tokens"] * self.input_price + usage["completion_tokens"] * self.output_price) / 1000000
        usage["token_price_per_million"] = {"input": self.input_price, "output": self.output_price}
        return usage


if __name__ == "__main__":
    t, u = chat([{"role": "user", "content": "Reply with just OK."}], max_tokens=8)
    print("response:", repr(t))
    print("usage:", u)
    print("counter:", get_counter())
