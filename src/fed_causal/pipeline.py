from __future__ import annotations
import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def identifier(value, context):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value).strip() == "":
        raise ValueError(f"{context} must be a nonempty string or integer")
    return str(value)

def number(value, context, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{context} must be a finite number >= {minimum}")
    return value

def read_modules(path):
    source = load_json(path)
    if isinstance(source, dict) and "modules" in source:
        source = source["modules"]
    if isinstance(source, list):
        mapped = {}
        for item in source:
            if not isinstance(item, dict) or "module_id" not in item:
                raise ValueError("Each module list entry must contain module_id")
            key = identifier(item["module_id"], "module_id")
            if key in mapped:
                raise ValueError(f"Duplicate module: {key}")
            mapped[key] = item
        source = mapped
    if not isinstance(source, dict) or len(source) < 2:
        raise ValueError("module-spec must contain at least two modules")
    modules = {}
    for key, value in source.items():
        key = identifier(key, "module_id")
        if not isinstance(value, dict):
            raise ValueError(f"Module {key} must be an object")
        fields = {}
        for name in ("X_vars", "A_vars", "I_out", "I_in"):
            items = value.get(name, [])
            if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
                raise ValueError(f"{key}.{name} must be a list of nonempty strings")
            if len(items) != len(set(items)):
                raise ValueError(f"{key}.{name} contains duplicate identifiers")
            fields[name] = items
        modules[key] = ModuleSpec(module_id=key, **fields)
    if not any(module.I_out for module in modules.values()) or not any(module.I_in for module in modules.values()):
        raise ValueError("module-spec requires outgoing and incoming interface event types")
    return modules, source

def read_trace_records(paths):
    records = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            value = load_json(path)
            values = value.get("traces", [value]) if isinstance(value, dict) else value
        if not isinstance(values, list) or not values:
            raise ValueError(f"No traces in {path}")
        for index, value in enumerate(values):
            if not isinstance(value, dict) or not isinstance(value.get("events"), list) or not value["events"]:
                raise ValueError(f"{path}: record {index} requires a nonempty events list")
            records.append((path, index, value))
    return records

def prepare_episodes(records, modules):
    episodes = {}
    raw_count = 0
    duplicate_count = 0
    for path, index, trace in records:
        episode_id = identifier(trace.get("episode_id", trace.get("trace_id")), "episode_id or trace_id")
        client_id = identifier(trace.get("client_id", "default"), "client_id")
        key = (client_id, episode_id)
        observed = trace.get("observed_modules", list(modules))
        if not isinstance(observed, list) or not observed:
            raise ValueError(f"{episode_id}: observed_modules must be a nonempty list")
        observed = [identifier(module, "observed_modules") for module in observed]
        if any(module not in modules for module in observed):
            raise ValueError(f"{episode_id}: observed_modules contains an unknown module")
        metadata = {name: trace.get(name) for name in ("task_id", "benchmark", "task_split")}
        for name, value in metadata.items():
            if value is not None:
                metadata[name] = identifier(value, name)
        if key not in episodes:
            episodes[key] = {"events": {}, "observed_modules": set(observed), "metadata": metadata, "declared_observation": "observed_modules" in trace}
        episode = episodes[key]
        for name, value in metadata.items():
            previous = episode["metadata"][name]
            if previous is not None and value is not None and previous != value:
                raise ValueError(f"{episode_id}: inconsistent {name}")
            if previous is None:
                episode["metadata"][name] = value
        episode["observed_modules"].update(observed)
        for row in trace["events"]:
            raw_count += 1
            if not isinstance(row, dict):
                raise ValueError(f"{episode_id}: every event must be an object")
            module = identifier(row.get("module_id"), "event.module_id")
            event_type = identifier(row.get("event_type"), "event.event_type")
            timestamp = number(row.get("timestamp"), "event.timestamp", -math.inf)
            payload = row.get("payload", {})
            if module not in modules or not isinstance(payload, dict):
                raise ValueError(f"{episode_id}: unknown module or invalid event payload")
            intervention = row.get("intervention_id")
            if intervention is not None:
                intervention = identifier(intervention, "intervention_id")
            event = Event(module, event_type, timestamp, payload, intervention)
            fingerprint = encoded({"module_id": module, "event_type": event_type, "timestamp": timestamp, "payload": payload, "intervention_id": intervention})
            event_id = row.get("event_id", payload.get("event_id"))
            dedup_key = (module, identifier(event_id, "event_id")) if event_id is not None else fingerprint
            if dedup_key in episode["events"]:
                if encoded(vars(episode["events"][dedup_key])) != fingerprint:
                    raise ValueError(f"{episode_id}: conflicting repeated event_id")
                duplicate_count += 1
            else:
                episode["events"][dedup_key] = event
    return episodes, raw_count, duplicate_count

def linked_to(event, intervention_id):
    candidates = [event.intervention_id, event.payload.get("intervention_id"), event.payload.get("response_to")]
    for candidate in candidates:
        if candidate is None:
            continue
        values = candidate if isinstance(candidate, list) else [candidate]
        if any(identifier(value, "response linkage") == intervention_id for value in values):
            return True
    return False

def discover(modules, episodes, args):
    evidence = defaultdict(lambda: {"opportunities": 0, "support": 0, "lags": [], "clients": set(), "direction_conflicts": 0})
    raw_edges = set()
    outgoing = sum(bool(module.I_out) for module in modules.values())
    trial_count = 0
    for (client_id, _), episode in episodes.items():
        events = sorted(episode["events"].values(), key=lambda event: event.timestamp)
        by_type = defaultdict(list)
        trials = {}
        for event in events:
            by_type[(event.module_id, event.event_type)].append(event)
            if event.event_type in modules[event.module_id].I_out and event.intervention_id is not None:
                key = (event.module_id, event.event_type, event.intervention_id)
                if key not in trials:
                    trials[key] = event
        trial_count += len(trials)
        for source in events:
            if source.event_type not in modules[source.module_id].I_out:
                continue
            for target_id, module in modules.items():
                if target_id == source.module_id:
                    continue
                for target_type in module.I_in:
                    if any(args.lag_min <= target.timestamp - source.timestamp <= args.lag_max for target in by_type[(target_id, target_type)]):
                        raw_edges.add((source.module_id, target_id, source.event_type, target_type))
        for (source_id, source_type, intervention_id), source in trials.items():
            for target_id in sorted(episode["observed_modules"]):
                if target_id == source_id:
                    continue
                for target_type in modules[target_id].I_in:
                    key = (source_id, target_id, source_type, target_type)
                    entry = evidence[key]
                    entry["opportunities"] += 1
                    entry["clients"].add(client_id)
                    responses = by_type[(target_id, target_type)]
                    if args.response_matching == "linked":
                        responses = [target for target in responses if linked_to(target, intervention_id)]
                    delays = [target.timestamp - source.timestamp for target in responses]
                    if args.response_matching == "linked" and any(delay < 0 for delay in delays):
                        entry["direction_conflicts"] += 1
                    matching = [delay for delay in delays if args.lag_min <= delay <= args.lag_max]
                    if matching:
                        entry["support"] += 1
                        entry["lags"].append(min(matching))
    if not trial_count:
        raise ValueError("No outgoing interface events contain intervention_id; intervention-response recovery requires actual intervention trials")
    if not evidence or not outgoing:
        raise ValueError("No observable cross-module interface opportunities")
    compact = []
    interfaces = []
    for (source, target, source_event, target_event), data in sorted(evidence.items()):
        support_rate = data["support"] / data["opportunities"]
        lags = data["lags"]
        entry = {"source": source, "target": target, "source_event": source_event, "target_event": target_event, "lag_min": min(lags) if lags else None, "lag_max": max(lags) if lags else None, "support": data["support"], "opportunities": data["opportunities"], "support_rate": support_rate, "direction_conflicts": data["direction_conflicts"], "clients": sorted(data["clients"])}
        compact.append(entry)
        if data["support"] >= args.min_support and support_rate >= args.min_response_rate and data["direction_conflicts"] <= args.max_direction_conflicts:
            interfaces.append(entry.copy())
    raw = [{"source": source, "target": target, "source_event": source_event, "target_event": target_event} for source, target, source_event, target_event in sorted(raw_edges)]
    return interfaces, compact, raw, trial_count

def oracle_metrics(path, modules, interfaces, raw_interfaces, granularity):
    document = load_json(path)
    rows = document.get("interfaces", document.get("edges")) if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("oracle-edges requires a list, or an object containing interfaces or edges")
    truth = set()
    for row in rows:
        if isinstance(row, dict):
            source, target = row.get("source"), row.get("target")
            source_event, target_event = row.get("source_event"), row.get("target_event")
        elif isinstance(row, list) and len(row) == 2 and granularity == "module":
            source, target = row
            source_event = target_event = None
        else:
            raise ValueError("oracle edge entries must contain source and target, plus event types for event granularity")
        source = identifier(source, "oracle.source")
        target = identifier(target, "oracle.target")
        if source not in modules or target not in modules or source == target:
            raise ValueError("oracle edge references unknown or identical modules")
        if granularity == "event":
            if source_event not in modules[source].I_out or target_event not in modules[target].I_in:
                raise ValueError("oracle edge has undeclared interface event types")
            truth.add((source, target, source_event, target_event))
        else:
            truth.add((source, target))
    metrics = {"granularity": granularity, "truth_edges": len(truth)}
    fields = ("source", "target", "source_event", "target_event") if granularity == "event" else ("source", "target")
    for name, predicted_rows in (("federated", interfaces), ("raw_temporal", raw_interfaces)):
        predicted = {tuple(row[field] for field in fields) for row in predicted_rows}
        tp = len(predicted & truth)
        fp = len(predicted - truth)
        fn = len(truth - predicted)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        metrics[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1}
    lhs, rhs = metrics["federated"]["f1"], metrics["raw_temporal"]["f1"]
    metrics["f1_improvement"] = lhs - rhs if lhs is not None and rhs is not None else None
    return metrics

def construction_usage(path):
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")
    if path is None:
        return {key: None for key in keys}
    source = load_json(path)
    rows = source.get("calls", [source]) if isinstance(source, dict) else source
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("construction-metadata must be a totals object or nonempty calls list")
    totals = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        for value in values:
            if value is not None:
                number(value, f"construction-metadata.{key}")
                if key != "cost_usd" and not isinstance(value, int):
                    raise ValueError(f"construction-metadata.{key} must be an integer")
        totals[key] = sum(values) if all(value is not None for value in values) else None
    if totals["total_tokens"] is None and all(totals[key] is not None for key in ("prompt_tokens", "completion_tokens")):
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals

def run(args):
    start = time.perf_counter()
    paths = sorted({path.resolve() for path in args.traces})
    modules, module_metadata = read_modules(args.module_spec)
    records = read_trace_records(paths)
    episodes, raw_count, duplicates = prepare_episodes(records, modules)
    interfaces, evidence, raw_interfaces, trials = discover(modules, episodes, args)
    construction_seconds = time.perf_counter() - start
    metadata = [episode["metadata"] for episode in episodes.values()]
    provenance = {"kind": "intervention_response_discovery", "response_matching": args.response_matching, "training_episode_ids": [{"client_id": client, "episode_id": episode} for client, episode in sorted(episodes)], "training_task_ids": sorted({item["task_id"] for item in metadata}) if all(item["task_id"] is not None for item in metadata) else None, "generated_at": datetime.now(timezone.utc).isoformat(), "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.module_spec.resolve(), *paths]}, "observation_policy": "observed_modules when declared; otherwise all modules are assumed observed within the episode", "support_unit": "unique (client_id, episode_id, source module, source event, intervention_id); earliest source and earliest matching response", "validation_rule": {"lag_min": args.lag_min, "lag_max": args.lag_max, "min_support": args.min_support, "min_response_rate": args.min_response_rate, "max_direction_conflicts": args.max_direction_conflicts}, "oracle_used_for_recovery": False, "interpretation": "Linkage records must identify intervention-response attribution; event timing alone does not establish direct causality."}
    for name in ("benchmark", "task_split"):
        values = {item[name] for item in metadata if item[name] is not None}
        if len(values) > 1:
            raise ValueError(f"Trace collection contains inconsistent {name} values")
        provenance[name] = next(iter(values)) if values and all(item[name] is not None for item in metadata) else None
    graph = {"modules": {key: {**module_metadata[key], **{name: getattr(value, name) for name in ("X_vars", "A_vars", "I_out", "I_in")}} for key, value in modules.items()}, "interfaces": interfaces, "provenance": provenance}
    raw_transport = encoded({"traces": [trace for _, _, trace in records]})
    compact_transport = encoded({"interfaces": evidence})
    graph_transport = encoded({"modules": list(modules), "interfaces": [{key: row[key] for key in ("source", "target", "source_event", "target_event", "lag_min", "lag_max", "support", "opportunities")} for row in interfaces]})
    raw_bytes, compact_bytes, graph_bytes = len(raw_transport), len(compact_transport), len(graph_transport)
    metrics = {"processed_trace_records": len(records), "processed_episodes": len(episodes), "raw_event_count": raw_count, "unique_event_count": raw_count - duplicates, "duplicate_event_count": duplicates, "intervention_trials": trials, "candidate_interfaces": len(raw_interfaces), "validated_interfaces": len(interfaces), "construction_seconds": construction_seconds, "construction_usage": construction_usage(args.construction_metadata), "communication": {"encoding": "UTF-8 JSON, sorted keys, no optional whitespace, uncompressed", "input_file_bytes": sum(path.stat().st_size for path in paths), "raw_trace_bytes": raw_bytes, "compact_evidence_bytes": compact_bytes, "interface_graph_bytes": graph_bytes, "raw_to_compact_evidence_ratio": raw_bytes / compact_bytes, "raw_to_graph_ratio": raw_bytes / graph_bytes, "compact_evidence_saved_percent": 100 * (1 - compact_bytes / raw_bytes), "graph_saved_percent": 100 * (1 - graph_bytes / raw_bytes), "traffic_scope": "Serialized payload sizes; excludes transport framing, retries, coordinator broadcast, and cryptographic overhead"}, "edge_evaluation": None}
    if args.oracle_edges:
        metrics["edge_evaluation"] = oracle_metrics(args.oracle_edges, modules, interfaces, raw_interfaces, args.edge_granularity)
        provenance["evaluation_oracle_sha256"] = hashlib.sha256(args.oracle_edges.read_bytes()).hexdigest()
    out = args.out_dir
    if out.exists():
        raise ValueError(f"Output directory already exists; refusing to overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)
    outputs = {"interfaces.json": encoded(graph), "raw_temporal_interfaces.json": encoded({"modules": graph["modules"], "interfaces": raw_interfaces, "provenance": {**provenance, "kind": "temporal_cooccurrence_baseline"}}), "compact_evidence.json": compact_transport, "interface_graph_transport.json": graph_transport, "raw_trace_transport.json": raw_transport, "metrics.json": encoded(metrics)}
    for name, contents in outputs.items():
        (out / name).write_bytes(contents)
    print(json.dumps({"out_dir": str(out), "validated_interfaces": len(interfaces), "processed_traces": len(records)}, sort_keys=True))

def construct_interfaces_main(argv=None):
    parser = argparse.ArgumentParser(description="Construct and evaluate modular interfaces from externally recorded intervention traces.")
    parser.add_argument("--module-spec", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True, help="JSON trace objects or JSONL trace records, each containing events and an episode_id or trace_id")
    parser.add_argument("--oracle-edges", type=Path, help="Held-out evaluation labels, read only after interface recovery")
    parser.add_argument("--construction-metadata", type=Path, help="Recorded construction prompt_tokens, completion_tokens, total_tokens, and cost_usd; unspecified fields remain null")
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / ("interface_construction_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--lag-min", type=float, default=0.0)
    parser.add_argument("--lag-max", type=float, default=5.0)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-response-rate", type=float, default=0.5)
    parser.add_argument("--max-direction-conflicts", type=int, default=0)
    parser.add_argument("--response-matching", choices=("linked", "temporal"), default="linked", help="linked requires target intervention_id or payload.intervention_id/response_to; temporal uses timing only")
    parser.add_argument("--edge-granularity", choices=("module", "event"), default="module")
    args = parser.parse_args(argv)
    if not math.isfinite(args.lag_min) or not math.isfinite(args.lag_max) or args.lag_min < 0 or args.lag_max < args.lag_min:
        parser.error("lag bounds must be finite and satisfy 0 <= lag-min <= lag-max")
    if args.min_support < 1 or not 0 <= args.min_response_rate <= 1 or args.max_direction_conflicts < 0:
        parser.error("min-support >= 1, min-response-rate in [0,1], and max-direction-conflicts >= 0 are required")
    try:
        run(args)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    if sys.argv[1:]:
        construct_interfaces_main()
    else:
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
