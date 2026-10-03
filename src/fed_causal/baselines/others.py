from typing import List, Tuple
from .base import BaselineBase, render_event_history, render_module_list
from event_traces import Task


class B0_NoWM(BaselineBase):
    name = "B0_NoWM"

    def build_prompt(self, task: Task) -> str:
        modules = render_module_list(task)
        last = task.events[-1] if task.events else None
        last_str = f"{last.module_id}.{last.event_type}" if last else "<none>"
        return (f"Modules: {modules}. Last event: {last_str}. "
                f"Predict next event. JSON: {{\"module_id\":...,\"event_type\":...}}")


class B1_LocalWM(BaselineBase):
    name = "B1_LocalWM"

    def build_prompt(self, task: Task) -> str:
        if not task.events:
            return ("Predict next event. JSON: {\"module_id\":...,\"event_type\":...}")
        active = task.events[-1].module_id
        local_history = [e for e in task.events if e.module_id == active][-10:]
        rendered = "\n".join(f"t={e.timestamp:>4} {e.module_id} {e.event_type}"
                             for e in local_history)
        return (f"You are the local world model for module '{active}'. Only its events:\n"
                f"{rendered}\n\nPredict next event. "
                f"JSON: {{\"module_id\":...,\"event_type\":...}}")


class B3_GlobalTransitionGraph(BaselineBase):
    name = "B3_GlobalTransitionGraph"

    def build_prompt(self, task: Task) -> str:
        h = render_event_history(task, max_events=30)
        return (f"Global transition graph over modular events:\n{h}\n\n"
                f"Predict next event from learned transition patterns.\n"
                f"JSON: {{\"module_id\":...,\"event_type\":...}}")


class B4_CorrelationWM(BaselineBase):
    name = "B4_CorrelationWM"

    def build_prompt(self, task: Task) -> str:
        h = render_event_history(task, max_events=30)
        return (f"Correlation-based dependency model. Recent events:\n{h}\n\n"
                f"Predict next event based on observed co-occurrence patterns.\n"
                f"JSON: {{\"module_id\":...,\"event_type\":...}}")


class B5_TemporalWM(BaselineBase):
    name = "B5_TemporalWM"

    def build_prompt(self, task: Task) -> str:
        h = render_event_history(task, max_events=30)
        return (f"Temporal world model. Pay attention to event ordering and lags:\n{h}\n\n"
                f"Predict next event. JSON: {{\"module_id\":...,\"event_type\":...}}")


class B6_InvariantWM(BaselineBase):
    name = "B6_InvariantWM"

    def build_prompt(self, task: Task) -> str:
        h = render_event_history(task, max_events=30)
        return (f"Invariant predictive WM: focus on policy-invariant features only.\n"
                f"History:\n{h}\n\n"
                f"Predict next event using only invariant patterns. "
                f"JSON: {{\"module_id\":...,\"event_type\":...}}")


class B7_CausalWMNoInt(BaselineBase):
    name = "B7_CausalWMNoInt"
    is_causal = True
    needs_intervention_data = False

    def __init__(self):
        from pipeline import (step1_local_modules, step2_interface_discovery,
                              step5_compose_graph, step5_topological_order)
        self._step1 = step1_local_modules
        self._step2 = step2_interface_discovery
        self._compose = step5_compose_graph
        self._topo = step5_topological_order

    def build_prompt(self, task: Task) -> str:
        from event_traces import BENCHMARKS
        spec, _, _ = BENCHMARKS[task.benchmark]
        modules = self._step1(spec)
        edges = self._step2(modules, task.events)
        G = self._compose(modules, edges)
        topo = " -> ".join(self._topo(G))
        h = render_event_history(task, max_events=30)
        return (f"Observational causal graph (topo order): {topo}\n"
                f"History:\n{h}\n\nPredict next event. "
                f"JSON: {{\"module_id\":...,\"event_type\":...}}")


class B8_FedCausalCompose(BaselineBase):
    name = "B8_FedCausalCompose"
    is_causal = True
    needs_intervention_data = True

    def __init__(self):
        from pipeline import run_fedcausalcompose
        self._run = run_fedcausalcompose

    def build_prompt(self, task: Task) -> str:
        from event_traces import BENCHMARKS
        spec, _, _ = BENCHMARKS[task.benchmark]
        res = self._run(spec, task.events, lag_window=5, p_verify_threshold=0.50)
        h = render_event_history(task, max_events=30)
        edges_str = ", ".join(f"{s}->{t}" for (s, _, t, _) in res.validated_edges[:20])
        topo = " -> ".join(res.topo_order)
        return (
            f"FedCausalCompose causal world model.\n"
            f"Validated cross-module edges (N_min={res.matcher_summary['N_min']}, "
            f"P_verify_min={res.matcher_summary['P_verify_min']:.3f}): {edges_str}\n"
            f"Topological order: {topo}\n"
            f"Event history:\n{h}\n\n"
            f"Apply causal rollout. Constraint: downstream module's incoming must be "
            f"justified by an upstream outgoing per the validated graph.\n"
            f"Predict next event. JSON: {{\"module_id\":...,\"event_type\":...}}"
        )


class B9_FCCNoControl(B8_FedCausalCompose):
    name = "B9_FCCNoControl"

    def build_prompt(self, task: Task) -> str:
        p = super().build_prompt(task)
        return p.replace(
            "Apply causal rollout. Constraint: downstream module's incoming must be "
            "justified by an upstream outgoing per the validated graph.\n", "")


class B10_OracleCausalWM(BaselineBase):
    name = "B10_OracleCausalWM"
    is_causal = True

    def build_prompt(self, task: Task) -> str:
        edges_str = ", ".join(f"{s}->{t}" for (s, _, t) in task.ground_truth_edges[:20])
        h = render_event_history(task, max_events=30)
        return (f"Oracle causal world model. GROUND-TRUTH edges: {edges_str}\n"
                f"History:\n{h}\n\n"
                f"Predict next event under causal rollout. "
                f"JSON: {{\"module_id\":...,\"event_type\":...}}")


class B11_CentralizedSeq(BaselineBase):
    name = "B11_CentralizedSeq"

    def build_prompt(self, task: Task) -> str:
        lines = []
        for ev in task.events:
            marker = "[do]" if ev.intervention_id else "    "
            lines.append(f"t={ev.timestamp:>4} {marker} {ev.module_id:<22} "
                         f"{ev.event_type:<28} payload={ev.payload}")
        return ("Centralized detailed sequence model. Full event log with payloads:\n"
                + "\n".join(lines)
                + "\n\nPredict next event. JSON: {\"module_id\":...,\"event_type\":...}")
