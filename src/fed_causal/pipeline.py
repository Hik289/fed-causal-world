from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

import networkx as nx



@dataclass
class ModuleSpec:
    module_id: str
    X_vars: List[str]
    A_vars: List[str]
    I_out: List[str]
    I_in: List[str]


@dataclass
class Event:
    module_id: str
    event_type: str
    timestamp: int
    payload: Dict[str, Any] = field(default_factory=dict)
    intervention_id: Optional[str] = None


@dataclass
class InterventionResponse:
    src_module: str
    src_event_type: str
    tgt_module: str
    tgt_event_type: str
    lag: int
    intervention_id: str



def step1_local_modules(modular_spec: Dict[str, Dict]) -> List[ModuleSpec]:
    return [ModuleSpec(
        module_id=mid,
        X_vars=list(spec.get("X_vars", [])),
        A_vars=list(spec.get("A_vars", [])),
        I_out=list(spec.get("I_out", [])),
        I_in=list(spec.get("I_in", [])),
    ) for mid, spec in modular_spec.items()]



def step2_interface_discovery(modules: List[ModuleSpec],
                              events: List[Event],
                              window: int = 5) -> List[Tuple[str, str, str, str]]:
    out_idx = defaultdict(list)
    in_idx = defaultdict(list)
    for ev in events:
        for m in modules:
            if m.module_id == ev.module_id and ev.event_type in m.I_out:
                out_idx[(m.module_id, ev.event_type)].append(ev)
            if m.module_id == ev.module_id and ev.event_type in m.I_in:
                in_idx[(m.module_id, ev.event_type)].append(ev)

    candidates: Set[Tuple[str, str, str, str]] = set()
    for (src_mod, src_ev), src_list in out_idx.items():
        for (tgt_mod, tgt_ev), tgt_list in in_idx.items():
            if src_mod == tgt_mod:
                continue
            j = 0
            for s in src_list:
                while j < len(tgt_list) and tgt_list[j].timestamp < s.timestamp:
                    j += 1
                if j < len(tgt_list) and tgt_list[j].timestamp - s.timestamp <= window:
                    candidates.add((src_mod, src_ev, tgt_mod, tgt_ev))
                    break
    return sorted(candidates)



class InterventionResponseMatcher:

    def __init__(self):
        self.matches: List[InterventionResponse] = []
        self.N_ij: Dict[Tuple[str, str], int] = defaultdict(int)
        self.q_count: Dict[str, int] = defaultdict(int)
        self.total_count: Dict[str, int] = defaultdict(int)
        self.r_obs: Dict[str, int] = defaultdict(int)
        self.r_pot: Dict[str, int] = defaultdict(int)

    def match(self, events: List[Event],
              candidate_edges: List[Tuple[str, str, str, str]],
              lag_window: int = 5) -> None:
        events_sorted = sorted(events, key=lambda e: e.timestamp)
        for ev in events_sorted:
            self.total_count[ev.module_id] += 1
            if ev.intervention_id is not None:
                self.q_count[ev.module_id] += 1

        edge_set = set((s, se, t, te) for (s, se, t, te) in candidate_edges)

        for i_idx, src in enumerate(events_sorted):
            if src.intervention_id is None:
                continue
            for tgt in events_sorted[i_idx + 1:]:
                dt = tgt.timestamp - src.timestamp
                if dt > lag_window:
                    break
                if (src.module_id, src.event_type, tgt.module_id, tgt.event_type) in edge_set:
                    self.r_pot[tgt.module_id] += 1
                    if dt >= 0:
                        self.r_obs[tgt.module_id] += 1
                        self.N_ij[(src.module_id, tgt.module_id)] += 1
                        self.matches.append(InterventionResponse(
                            src_module=src.module_id,
                            src_event_type=src.event_type,
                            tgt_module=tgt.module_id,
                            tgt_event_type=tgt.event_type,
                            lag=dt,
                            intervention_id=src.intervention_id,
                        ))

    def q_hat(self) -> Dict[str, float]:
        return {k: (self.q_count[k] / self.total_count[k] if self.total_count[k] else 0.0)
                for k in self.total_count}

    def r_hat(self) -> Dict[str, float]:
        return {k: (self.r_obs[k] / self.r_pot[k] if self.r_pot[k] else 0.0)
                for k in self.r_pot}

    def N_min(self, true_edges: Optional[List[Tuple[str, str]]] = None) -> int:
        if true_edges:
            edges = true_edges
        else:
            edges = list(self.N_ij.keys())
        if not edges:
            return 0
        return min(self.N_ij.get(e, 0) for e in edges)

    def p_verify(self, q_min: Optional[float] = None,
                 r_min: Optional[float] = None,
                 N_ij: Optional[int] = None) -> float:
        if q_min is None:
            q_hat = self.q_hat()
            q_min = min(q_hat.values()) if q_hat else 0.0
        if r_min is None:
            r_hat = self.r_hat()
            r_min = min(r_hat.values()) if r_hat else 0.0
        if N_ij is None:
            N_ij = self.N_min()
        return 1.0 - (1.0 - q_min * r_min) ** max(N_ij, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_matches": len(self.matches),
            "N_ij": {f"{i}->{j}": n for (i, j), n in self.N_ij.items()},
            "q_hat": self.q_hat(),
            "r_hat": self.r_hat(),
            "N_min": self.N_min(),
            "P_verify_min": self.p_verify(),
        }


def step3_distributed_intervention_matching(
        events: List[Event],
        candidate_edges: List[Tuple[str, str, str, str]],
        lag_window: int = 5) -> InterventionResponseMatcher:
    m = InterventionResponseMatcher()
    m.match(events, candidate_edges, lag_window=lag_window)
    return m



def step4_edge_validation(
        matcher: InterventionResponseMatcher,
        candidate_edges: List[Tuple[str, str, str, str]],
        p_verify_threshold: float = 0.50) -> List[Tuple[str, str, str, str]]:
    q_hat = matcher.q_hat()
    r_hat = matcher.r_hat()
    validated = []
    for (s, se, t, te) in candidate_edges:
        n = matcher.N_ij.get((s, t), 0)
        q = q_hat.get(s, 0.0)
        r = r_hat.get(t, 0.0)
        pv = 1.0 - (1.0 - q * r) ** max(n, 0)
        if pv >= p_verify_threshold:
            validated.append((s, se, t, te))
    return validated



def step5_compose_graph(modules: List[ModuleSpec],
                        validated_edges: List[Tuple[str, str, str, str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for m in modules:
        G.add_node(m.module_id)
    for (s, _, t, _) in validated_edges:
        G.add_edge(s, t)
    return G


def step5_topological_order(G: nx.DiGraph) -> List[str]:
    if not nx.is_directed_acyclic_graph(G):
        H = G.copy()
        while not nx.is_directed_acyclic_graph(H):
            bet = nx.edge_betweenness_centrality(H)
            worst = max(bet, key=bet.get)
            H.remove_edge(*worst)
        return list(nx.topological_sort(H))
    return list(nx.topological_sort(G))



def step6_emit_constraints(G: nx.DiGraph, module_id: str) -> Dict[str, List[str]]:
    upstream = list(G.predecessors(module_id))
    downstream = list(G.successors(module_id))
    return {
        "module_id": module_id,
        "upstream_required": upstream,
        "downstream_predict": downstream,
        "block_if_violates_global_constraint": True,
        "verify_after_execution": True,
    }



@dataclass
class FCCResult:
    candidate_edges: List[Tuple[str, str, str, str]]
    validated_edges: List[Tuple[str, str, str, str]]
    matcher_summary: Dict[str, Any]
    topo_order: List[str]
    constraints_per_module: Dict[str, Dict[str, List[str]]]


def run_fedcausalcompose(modular_spec: Dict[str, Dict],
                         events: List[Event],
                         lag_window: int = 5,
                         p_verify_threshold: float = 0.50) -> FCCResult:
    modules = step1_local_modules(modular_spec)
    candidate_edges = step2_interface_discovery(modules, events, window=lag_window)
    matcher = step3_distributed_intervention_matching(events, candidate_edges, lag_window)
    validated = step4_edge_validation(matcher, candidate_edges, p_verify_threshold)
    G = step5_compose_graph(modules, validated)
    order = step5_topological_order(G)
    constraints = {m.module_id: step6_emit_constraints(G, m.module_id) for m in modules}
    return FCCResult(
        candidate_edges=candidate_edges,
        validated_edges=validated,
        matcher_summary=matcher.to_dict(),
        topo_order=order,
        constraints_per_module=constraints,
    )



if __name__ == "__main__":
    spec = {
        "A": {"X_vars": ["x"], "A_vars": ["do_a"], "I_out": ["a_done"], "I_in": []},
        "B": {"X_vars": ["y"], "A_vars": ["do_b"], "I_out": ["b_done"], "I_in": ["a_done"]},
        "C": {"X_vars": ["z"], "A_vars": ["do_c"], "I_out": [], "I_in": ["b_done"]},
    }
    events = []
    t = 0
    for i in range(20):
        events.append(Event("A", "a_done", t,   intervention_id=f"int_{i}"))
        events.append(Event("B", "a_done", t + 1))
        events.append(Event("B", "b_done", t + 2))
        events.append(Event("C", "b_done", t + 3))
        t += 5
    res = run_fedcausalcompose(spec, events)
    print(json.dumps({
        "candidate_edges": res.candidate_edges,
        "validated_edges": res.validated_edges,
        "matcher": res.matcher_summary,
        "topo_order": res.topo_order,
    }, indent=2))
