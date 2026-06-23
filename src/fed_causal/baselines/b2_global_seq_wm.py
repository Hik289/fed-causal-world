"""
B2: Global Sequence World Model — concatenates ALL modules' events as a global
time-ordered sequence and asks the LLM to predict next event. NO causal graph,
NO intervention semantics. The single most important non-causal baseline.

Reference: Wang et al. 2024 (LLMs as text-based world simulators) + DyMo 2025.
"""

from .base import BaselineBase, render_event_history, render_module_list
from event_traces import Task


class B2_GlobalSeqWM(BaselineBase):
    name = "B2_GlobalSeqWM"
    is_causal = False
    needs_intervention_data = False

    def build_prompt(self, task: Task) -> str:
        history = render_event_history(task)
        modules = render_module_list(task)
        prompt = (
            f"You are observing a modular system with the following modules: {modules}.\n"
            f"Global time-ordered event history:\n"
            f"{history}\n\n"
            f"Predict the SINGLE next event that will occur.\n"
            f"Output strictly as JSON: {{\"module_id\": \"...\", \"event_type\": \"...\"}}"
        )
        return prompt
