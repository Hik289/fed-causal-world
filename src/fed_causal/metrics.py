"""
metrics.py — Evaluation metrics for fedcausalworld experiments.

Implements (per FedCausal.md §16):
  - Transition EM    : exact match of (module_id, event_type) prediction
  - StateAcc         : same as Transition EM for the synthetic event-prediction
                       task (no separate state vector in the dry-run harness;
                       full state-vector StateAcc applies to Synthetic SCM)
  - Edge F1, Direction Accuracy, Lag Accuracy : §16.5 structure diagnostics

Reporting helpers:
  - aggregate_per_module : per-module EM, required by EDA insight §4.2 (do NOT
                            use global EM only — confounder bias is diluted).
  - bootstrap_ci         : 95% bootstrap CI half-width over task list.
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple, Any


def transition_em(pred: Dict[str, Any], target: Dict[str, Any]) -> int:
    return int(pred.get("module_id") == target.get("module_id")
               and pred.get("event_type") == target.get("event_type"))


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """records: list of {pred, target, task_id, benchmark, usage}."""
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
    """Edge F1 over (src_module, tgt_module) pairs."""
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
