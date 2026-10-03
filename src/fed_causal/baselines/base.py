from __future__ import annotations
import json
from typing import Dict, Any, Tuple
from event_traces import Task
from llm_client import chat as llm_chat


class BaselineBase:
    name = "base"
    is_causal = False
    needs_intervention_data = False

    def build_prompt(self, task: Task) -> str:
        raise NotImplementedError

    def predict_next_event(self, task: Task,
                           max_output_tokens: int = 96) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self.build_prompt(task)
        messages = [
            {"role": "system",
             "content": "You are a world model. Given an event history of a modular system, "
                        "predict the next event in strict JSON. Output ONLY one JSON object "
                        "with keys: module_id, event_type. No extra text."},
            {"role": "user", "content": prompt},
        ]
        text, usage = llm_chat(messages, max_tokens=max_output_tokens, temperature=0.0)
        pred = _parse_event_json(text)
        return pred, usage


def _parse_event_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {"module_id": None, "event_type": None, "raw": text}
    try:
        obj = json.loads(text[s:e + 1])
        return {
            "module_id": obj.get("module_id") or obj.get("module"),
            "event_type": obj.get("event_type") or obj.get("event"),
            "raw": text,
        }
    except Exception as exc:
        return {"module_id": None, "event_type": None, "raw": text, "parse_error": str(exc)}


def render_event_history(task: Task, max_events: int = 60) -> str:
    lines = []
    for i, ev in enumerate(task.events[-max_events:]):
        marker = "[do]" if ev.intervention_id else "    "
        lines.append(f"t={ev.timestamp:>4} {marker} {ev.module_id:<22} {ev.event_type}")
    return "\n".join(lines)


def render_module_list(task: Task) -> str:
    from event_traces import BENCHMARKS
    spec, _, _ = BENCHMARKS[task.benchmark]
    return ", ".join(spec.keys())
