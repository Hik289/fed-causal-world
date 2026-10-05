from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import math
import random
from typing import Dict, List, Tuple, Any


def transition_em(pred: Dict[str, Any], target: Dict[str, Any]) -> int:
    return int(pred.get("module_id") == target.get("module_id")
               and pred.get("event_type") == target.get("event_type"))


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}
    em = [transition_em(r["pred"], r["target"]) for r in records]
    by_mod_em = {}
    by_mod_n = {}
    for r, e in zip(records, em):
        m = r["target"].get("module_id") or "?"
        by_mod_em[m] = by_mod_em.get(m, 0) + e
        by_mod_n[m] = by_mod_n.get(m, 0) + 1
    per_mod = {m: by_mod_em[m] / by_mod_n[m] for m in by_mod_em}
    return {
        "n": len(records),
        "transition_em": sum(em) / len(em),
        "transition_em_95ci_half": bootstrap_ci_halfwidth(em),
        "per_module_em": per_mod,
        "module_id_match": sum(1 for r in records
                               if r["pred"].get("module_id") == r["target"].get("module_id")) / len(records),
        "event_type_match": sum(1 for r in records
                                if r["pred"].get("event_type") == r["target"].get("event_type")) / len(records),
        "parse_success_rate": sum(1 for r in records if r["pred"].get("event_type")) / len(records),
    }


def edge_f1(predicted_edges: List[Tuple[str, str]],
            true_edges: List[Tuple[str, str]]) -> Dict[str, float]:
    P = set(predicted_edges)
    T = set(true_edges)
    tp = len(P & T)
    fp = len(P - T)
    fn = len(T - P)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def bootstrap_ci_halfwidth(values: List[float], n_resamples: int = 1000,
                           alpha: float = 0.05, seed: int = 0) -> float:
    if not values:
        return 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        samp = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(samp) / n)
    means.sort()
    lo = means[int(n_resamples * (alpha / 2))]
    hi = means[int(n_resamples * (1 - alpha / 2))]
    return (hi - lo) / 2.0


METRICS = ("score", "success_percent", "logical_calls", "provider_request_attempts", "tokens", "elapsed_s", "cost_usd", "simulator_calls", "simulator_tokens", "simulator_cost_usd")

EXCLUDED_SETTINGS = {"condition", "graph", "model", "models", "seed", "seeds", "attention_anchor", "immediate_feedback", "control", "out_dir", "outdir", "task_indices", "n_tasks", "task_manifest", "dialogue_manifest", "alfworld_config", "dataset", "data_dir", "dataset_dir", "benchmark_dir", "apibank_root", "jar_path"}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def numeric(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or null")
    return value

def metric_values(record, benchmark):
    usage = record.get("usage") or {}
    simulator = record.get("simulator_usage") or {}
    success = record.get("success")
    if success is not None and not isinstance(success, bool):
        raise ValueError("success must be boolean or null")
    values = {"score": record.get("call_accuracy") if benchmark == "apibank" else record.get("score"),
              "success_percent": 100 * int(success) if success is not None else None,
              "logical_calls": usage.get("calls"), "provider_request_attempts": usage.get("provider_request_attempts"),
              "tokens": usage.get("total_tokens"), "elapsed_s": record.get("elapsed_s", record.get("wall_time_s")),
              "cost_usd": usage.get("estimated_cost_usd"), "simulator_calls": simulator.get("calls"),
              "simulator_tokens": simulator.get("total_tokens"), "simulator_cost_usd": simulator.get("estimated_cost_usd")}
    return {name: numeric(value, name) for name, value in values.items()}

def load_run(path, label):
    modular = (path / "manifest.json").is_file()
    meta_path = path / ("manifest.json" if modular else "configuration.json")
    metadata = read_json(meta_path)
    settings = metadata.get("config" if modular else "arguments")
    if not isinstance(settings, dict):
        raise ValueError(f"Missing configuration arguments in {path}")
    benchmark = metadata.get("benchmark", settings.get("benchmark"))
    if benchmark is None and "scienceworld" in str(metadata.get("source_api", "")).lower():
        benchmark = "scienceworld"
    if benchmark not in ("alfworld", "tau_retail", "tau_airline", "scienceworld", "apibank"):
        raise ValueError(f"Unknown benchmark in {path}: {benchmark}")
    tasks = metadata.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"A nonempty task manifest is required in {path}")
    task_map = {}
    for task in tasks:
        task_id = task.get("task_id") if isinstance(task, dict) else None
        if not isinstance(task_id, str) or not task_id or task_id in task_map:
            raise ValueError(f"Invalid or repeated task_id in {path}")
        task_map[task_id] = {key: value for key, value in task.items() if key not in ("index", "game_file", "source_path", "file_path", "task_source_file")}
    data_path = path / ("records.json" if modular else "episodes.jsonl")
    if data_path.exists():
        records = read_json(data_path) if modular else [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = []
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Invalid episode records in {path}")
    protocol = {"benchmark": benchmark, "runner_format": "modular" if modular else "episodes", "protocol": metadata.get("protocol"),
                "settings": {key: value for key, value in settings.items() if key not in EXCLUDED_SETTINGS},
                "versions": metadata.get("versions", {"scienceworld": metadata.get("scienceworld_version")}),
                "dataset_sha256": metadata.get("dataset_sha256") if benchmark != "apibank" else None,
                "task_source_sha256": metadata.get("task_source_sha256"),
                "jar_sha256": metadata.get("simulator_jar_sha256", metadata.get("jar_sha256")), "alfworld_config_sha256": metadata.get("alfworld_config_sha256"),
                "tool_manager_sha256": metadata.get("tool_manager_sha256"), "api_source_sha256": metadata.get("api_source_sha256"),
                "initial_database_sha256": metadata.get("initial_database_sha256"),
                "simulator_model": metadata.get("simulator_model"), "simulator_provider": metadata.get("simulator_provider")}
    if not protocol["protocol"]:
        raise ValueError(f"Missing evaluation protocol in {path}")
    models = metadata.get("models") or [settings.get("model")]
    seeds = metadata.get("seeds") or [settings.get("seed")]
    if any(not isinstance(model, str) or not model for model in models) or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError(f"Invalid model or seed metadata in {path}")
    condition = settings.get("condition")
    if not isinstance(condition, str) or not condition:
        raise ValueError(f"Missing condition in {path}")
    selectors = {"condition": condition, "attention_anchor": settings.get("attention_anchor", False),
                 "immediate_feedback": settings.get("immediate_feedback", False),
                 "control": settings.get("control", True) and condition == "causalcompose", "graph_sha256": metadata.get("graph_sha256")}
    normalized = []
    identities = set()
    for record in records:
        task_id = record.get("task_id")
        seed = record.get("seed", settings.get("seed"))
        model = record.get("model", settings.get("model"))
        if task_id not in task_map or seed not in seeds or model not in models:
            raise ValueError(f"Record identity is absent from its manifest in {path}")
        if record.get("benchmark", benchmark) != benchmark:
            raise ValueError(f"Record benchmark disagrees with its manifest in {path}")
        if "graph_sha256" in record and record["graph_sha256"] != selectors["graph_sha256"]:
            raise ValueError(f"Record graph fingerprint disagrees with its manifest in {path}")
        for key in ("condition", "attention_anchor", "immediate_feedback", "control"):
            if key in record and record[key] != selectors[key]:
                raise ValueError(f"Record {key} disagrees with its configuration in {path}")
        identity = (model, seed, task_id)
        if identity in identities:
            raise ValueError(f"Duplicate task/model/seed pair in {path}: {identity}")
        identities.add(identity)
        status = record.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"Missing record status in {path}")
        complete = status in ("ok", "completed")
        values = metric_values(record, benchmark)
        if complete and values["score"] is None:
            raise ValueError(f"Completed record has no benchmark score in {path}: {task_id}")
        normalized.append({"model": model, "seed": seed, "task_id": task_id, "status": status, "complete": complete,
                           "metrics": values, "error_type": record.get("error_type"),
                           "correct_calls": numeric(record.get("correct_calls"), "correct_calls"),
                           "total_calls": numeric(record.get("total_calls"), "total_calls")})
    return {"path": str(path), "label": label, "benchmark": benchmark, "protocol": protocol,
            "protocol_id": fingerprint(protocol), "selectors": selectors, "models": models, "seeds": seeds,
            "tasks": task_map, "records": normalized,
            "setup_summary": read_json(path / "summary.json") if not records and (path / "summary.json").exists() else None,
            "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest()}

def mean_metric(rows, name):
    values = [row["metrics"][name] for row in rows]
    observed = [value for value in values if value is not None]
    return {"n": len(observed), "denominator": len(rows), "mean": statistics.mean(observed) if observed and len(observed) == len(values) else None}

def seed_summary(rows, expected):
    completed = [row for row in rows if row["complete"]]
    result = {"expected": expected, "attempted": len(rows), "completed": len(completed),
              "errors": len(rows) - len(completed), "missing": expected - len(rows),
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "completed_metrics": {name: mean_metric(completed, name) for name in METRICS},
              "all_attempt_metrics": {name: mean_metric(rows, name) for name in METRICS if name not in ("score", "success_percent")}}
    call_counts = [(row["correct_calls"], row["total_calls"]) for row in completed]
    if call_counts and all(correct is not None and total is not None for correct, total in call_counts):
        correct = sum(pair[0] for pair in call_counts)
        total = sum(pair[1] for pair in call_counts)
        result["micro_call_accuracy"] = {"correct_calls": correct, "total_calls": total, "value": correct / total if total else None}
    return result

def across_seeds(per_seed, field):
    result = {}
    names = METRICS if field == "completed_metrics" else tuple(name for name in METRICS if name not in ("score", "success_percent"))
    for name in names:
        values = [seed[field][name]["mean"] for seed in per_seed]
        valid = [value for value in values if value is not None]
        available = bool(valid) and len(valid) == len(values)
        result[name] = {"seeds_available": len(valid), "seeds_expected": len(values),
                        "mean": statistics.mean(valid) if available else None,
                        "std": statistics.stdev(valid) if available and len(valid) > 1 else None}
    return result

def group_runs(runs):
    groups = {}
    for run in runs:
        for model in run["models"]:
            configuration = {"benchmark": run["benchmark"], "protocol_id": run["protocol_id"], "model": model, **run["selectors"]}
            group_id = fingerprint(configuration)
            group = groups.setdefault(group_id, {"group_id": group_id, "configuration": configuration, "protocol": run["protocol"],
                                                 "run_paths": [], "labels": [], "records": {}, "expected": set(), "tasks": {}})
            group["run_paths"].append(run["path"])
            group["labels"].append(run["label"])
            for task_id, identity in run["tasks"].items():
                if task_id in group["tasks"] and group["tasks"][task_id] != identity:
                    raise ValueError(f"Conflicting dataset task content for {task_id}")
                group["tasks"][task_id] = identity
                for seed in run["seeds"]:
                    key = (seed, task_id)
                    if key in group["expected"]:
                        raise ValueError(f"Repeated task/seed assignment within the same model and condition: {key}")
                    group["expected"].add(key)
            for record in run["records"]:
                if record["model"] == model:
                    group["records"][(record["seed"], record["task_id"])] = record
    return list(groups.values())

def summarize_group(group):
    seeds = sorted({seed for seed, _ in group["expected"]})
    per_seed = [{"seed": seed, **seed_summary([row for (row_seed, _), row in group["records"].items() if row_seed == seed],
                                             sum(expected_seed == seed for expected_seed, _ in group["expected"]))} for seed in seeds]
    return {key: value for key, value in group.items() if key not in ("records", "expected", "tasks")} | {
        "per_seed": per_seed, "completed_metrics_across_seeds": across_seeds(per_seed, "completed_metrics"),
        "all_attempt_metrics_across_seeds": across_seeds(per_seed, "all_attempt_metrics")}

def compare_groups(left, right):
    common = left["expected"] & right["expected"]
    for _, task_id in common:
        if left["tasks"][task_id] != right["tasks"][task_id]:
            raise ValueError(f"Paired task {task_id} has conflicting source identities")
    both_present = common & left["records"].keys() & right["records"].keys()
    complete = {key for key in both_present if left["records"][key]["complete"] and right["records"][key]["complete"]}
    paired_rows = []
    for key in sorted(both_present):
        a, b = left["records"][key], right["records"][key]
        deltas = {name: b["metrics"][name] - a["metrics"][name] if a["metrics"][name] is not None and b["metrics"][name] is not None else None for name in METRICS}
        paired_rows.append({"seed": key[0], "task_id": key[1], "complete": key in complete,
                            "status": "paired_completed" if key in complete else "paired_error", "metrics": deltas, "correct_calls": None, "total_calls": None})
    per_seed = [{"seed": seed, **seed_summary([row for row in paired_rows if row["seed"] == seed], sum(key[0] == seed for key in common))} for seed in sorted({key[0] for key in common})]
    return {"left_group": left["group_id"], "right_group": right["group_id"], "delta_direction": "right minus left",
            "shared_expected_pairs": len(common), "shared_attempted_pairs": len(both_present), "paired_completed": len(complete),
            "pairs_with_error": len(both_present - complete), "pairs_missing_record": len(common - both_present),
            "left_only_assignments": len(left["expected"] - right["expected"]), "right_only_assignments": len(right["expected"] - left["expected"]),
            "per_seed_differences": per_seed, "paired_metrics_across_seeds": across_seeds(per_seed, "completed_metrics"),
            "paired_units": [{"seed": row["seed"], "task_id": row["task_id"], "status": row["status"], "differences": row["metrics"]} for row in paired_rows]}

def summarize_agent_runs_main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate existing agent evaluation records without running experiments or making API calls.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = [path.resolve() for path in args.runs]
        if len(set(paths)) != len(paths):
            raise ValueError("Run directories must be unique")
        labels = args.labels or [path.name for path in paths]
        if len(labels) != len(paths):
            raise ValueError("--labels must have one entry per --runs directory")
        if args.out_dir.exists():
            raise ValueError(f"Refusing to overwrite existing output directory: {args.out_dir}")
        runs = [load_run(path, label) for path, label in zip(paths, labels)]
        groups = group_runs(runs)
        comparisons = []
        skipped = []
        for left, right in itertools.combinations(groups, 2):
            if left["configuration"]["protocol_id"] != right["configuration"]["protocol_id"]:
                skipped.append({"left_group": left["group_id"], "right_group": right["group_id"], "reason": "different benchmark, dataset identity, configuration, or evaluation protocol"})
            else:
                comparisons.append(compare_groups(left, right))
        result = {"aggregation": "Equal-weight mean and sample standard deviation of per-seed task means; missing measurements propagate null. Errors are excluded from completed metrics and retained in all-attempt resource metrics.",
                  "score_semantics": {"alfworld": "environment reward", "tau_retail": "official terminal reward", "tau_airline": "official terminal reward", "scienceworld": "environment score (0-100)", "apibank": "per-dialogue call accuracy (0-1); success requires every target call correct"},
                  "cost_scope": "Agent and simulator costs remain separate; absent prices or incomplete usage produce null.",
                  "runs": [{key: run[key] for key in ("path", "label", "metadata_sha256", "setup_summary")} for run in runs],
                  "groups": [summarize_group(group) for group in groups], "paired_comparisons": comparisons, "skipped_comparisons": skipped}
        contents = canonical(result)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        (args.out_dir / "agent_comparison_summary.json").write_text(contents + "\n", encoding="utf-8")
        print(json.dumps({"out_dir": str(args.out_dir), "groups": len(groups), "paired_comparisons": len(comparisons)}))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__" and sys.argv[1:]:
    summarize_agent_runs_main()
