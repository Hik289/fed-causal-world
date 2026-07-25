"""
anchor4_sanity.py — H0.anchor_4 Synthetic Modular SCM Sanity Gate.

Implements B2 / B7 / B8 / B10 as numerical predictors on the Synthetic SCM:
  - B10 Oracle Causal WM       : uses ground-truth W_self, B, W_cross
  - B8  FedCausalCompose       : recovers GM* via Step 3 (intervention matching)
                                  + Step 4 (P_verify threshold), fits OLS per
                                  recovered parent set
  - B7  Causal WM no-int       : uses only observational candidate edges
                                  (skips Step 3-4 verification), fits OLS
  - B2  Global Sequence WM     : single global OLS on concat(X_t, A_t)→X_{t+1},
                                  NO graph structure assumed

Evaluation: Transition EM on the unseen_int_test split (intervention generalization
is where causal advantage should appear per Theorem 1 lower bound; on the IID
obs split alone, all linear OLS variants can fit well).

Discretization: per-variable 10-bin percentile from the train split. EM = mean
of "predicted bin == true bin" across (t, k, q).

Theorem 2 verification: track N_min and compute the empirical N* needed to
reach P_verify >= 0.95 per equation (eq:p-verify).
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_split(data_dir: str, split_name: str) -> Dict[str, np.ndarray]:
    path = os.path.join(data_dir, f"{split_name}.npz")
    return dict(np.load(path, allow_pickle=False))


def load_oracle(data_dir: str) -> Dict[str, Any]:
    with open(os.path.join(data_dir, "oracle.pkl"), "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# Discretization
# ----------------------------------------------------------------------

def fit_bin_edges(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """X: (T, K, n_k). Return edges (K, n_k, n_bins+1) using train-split quantiles."""
    T, K, n = X.shape
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.empty((K, n, n_bins + 1))
    for k in range(K):
        for q in range(n):
            edges[k, q] = np.quantile(X[:, k, q], qs)
            # ensure monotone (jitter ties)
            for i in range(1, n_bins + 1):
                if edges[k, q, i] <= edges[k, q, i - 1]:
                    edges[k, q, i] = edges[k, q, i - 1] + 1e-6
    return edges


def discretize(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    T, K, n = X.shape
    out = np.empty((T, K, n), dtype=np.int8)
    n_bins = edges.shape[-1] - 1
    for k in range(K):
        for q in range(n):
            out[:, k, q] = np.clip(
                np.searchsorted(edges[k, q, 1:-1], X[:, k, q]),
                0, n_bins - 1,
            ).astype(np.int8)
    return out


def transition_em(pred_disc: np.ndarray, true_disc: np.ndarray) -> float:
    return float((pred_disc == true_disc).mean())


def per_module_em(pred_disc: np.ndarray, true_disc: np.ndarray) -> Dict[int, float]:
    K = pred_disc.shape[1]
    return {k: float((pred_disc[:, k] == true_disc[:, k]).mean()) for k in range(K)}


# ----------------------------------------------------------------------
# B10: Oracle Causal WM
# ----------------------------------------------------------------------

def b10_predict(X_obs: np.ndarray, A_obs: np.ndarray, oracle: Dict[str, Any],
                K: int, n: int) -> np.ndarray:
    """Use ground-truth W_self, B, W_cross to predict X_{t+1}.

    For each module k:
       X_{t+1, k} = W_self_k @ X_{t, k} + B_k @ A_{t+1, k}
                  + alpha * sum_{(i,k,lag) in W_cross} W_ik @ tanh(X_{t-lag, i})

    (gamma = 0 in sanity_4mod → no confounder term.)
    """
    params = oracle["mechanism_params"]
    cfg = oracle["config"]
    alpha = cfg["alpha"]
    GV_edges = oracle["GV_edges"]

    T = X_obs.shape[0]
    pred = np.zeros_like(X_obs)
    # Note: action timing convention in synthetic_scm_skeleton.py is that
    # A[t] drives X[t] (see simulate loop). So predicting X[t+1] uses A[t+1].
    for t in range(1, T):
        for k in range(K):
            x_next = params["W_self"][k] @ X_obs[t - 1, k] + params["B"][k] @ A_obs[t, k]
            for (i, p, j, q, lag) in GV_edges:
                if j != k:
                    continue
                t_src = max(0, t - 1 - lag)
                signal = params["W_cross"][(i, j, lag)] @ np.tanh(X_obs[t_src, i])
                x_next += alpha * signal
            pred[t, k] = x_next
    pred[0] = X_obs[0]  # no prior info
    return pred


# ----------------------------------------------------------------------
# B2: Global Sequence WM (single big OLS, NO graph)
# ----------------------------------------------------------------------

def b2_fit(X_train: np.ndarray, A_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit X_{t+1} = W_global @ concat(X_t, A_{t+1}) + b_global by OLS.

    Returns W_global (K*n, K*n + K*m) and b_global (K*n,).
    """
    T, K, n = X_train.shape
    _, _, m = A_train.shape
    # design matrix: (T-1, K*n + K*m), target: (T-1, K*n)
    Xt = X_train[:-1].reshape(T - 1, K * n)
    At = A_train[1:].reshape(T - 1, K * m)
    Xtp1 = X_train[1:].reshape(T - 1, K * n)
    Phi = np.concatenate([Xt, At, np.ones((T - 1, 1))], axis=1)  # +bias
    # Solve OLS in closed form
    W_full, *_ = np.linalg.lstsq(Phi, Xtp1, rcond=None)
    return W_full.T, K * n + K * m  # (K*n, P), P


def b2_predict(X_obs: np.ndarray, A_obs: np.ndarray, W_full: np.ndarray,
               K: int, n: int, m: int) -> np.ndarray:
    T = X_obs.shape[0]
    Xt = X_obs[:-1].reshape(T - 1, K * n)
    At = A_obs[1:].reshape(T - 1, K * m)
    Phi = np.concatenate([Xt, At, np.ones((T - 1, 1))], axis=1)
    Xtp1_flat = Phi @ W_full.T
    pred = np.zeros_like(X_obs)
    pred[1:] = Xtp1_flat.reshape(T - 1, K, n)
    pred[0] = X_obs[0]
    return pred


# ----------------------------------------------------------------------
# Step 3 + 4 (numerical): edge recovery from synthetic intervention data
# ----------------------------------------------------------------------

def synthetic_step3_step4(X_train: np.ndarray, A_train: np.ndarray,
                          int_mask: np.ndarray, K: int, n: int,
                          lag_window: int = 3,
                          p_verify_threshold: float = 0.50) -> Dict[str, Any]:
    """Compute per-pair (i, j) N_ij, q_i, r_j and apply P_verify >= threshold.

    Implementation strategy (matches pipeline.py InterventionResponseMatcher):
      - q_i  := mean(int_mask[:, i])
      - For each timestep t where module i had intervention, count downstream
        responses in module j within [t+1, t+lag_window] (proxy: nontrivial
        change in X[:, j]); r_j averaged across (i, j, t).
      - N_ij := number of (t, i, j) triples where the response is detected.
    """
    T = X_train.shape[0]
    q_hat = np.array([int_mask[:, i].mean() for i in range(K)])

    # Detect "change" via z-score of dX
    dX = np.linalg.norm(X_train[1:] - X_train[:-1], axis=2)   # (T-1, K)
    dX_thresh = np.quantile(dX, 0.50)   # change if dX > median across all (t,k)

    N_ij = np.zeros((K, K), dtype=int)
    r_pot = np.zeros(K, dtype=int)
    r_obs = np.zeros(K, dtype=int)
    for t in range(T - lag_window):
        for i in range(K):
            if int_mask[t, i] == 0:
                continue
            for j in range(K):
                if i == j:
                    continue
                r_pot[j] += 1
                # check any response in [t+1, t+lag_window]
                window_change = (dX[t:t + lag_window, j] > dX_thresh).any()
                if window_change:
                    r_obs[j] += 1
                    N_ij[i, j] += 1

    r_hat = np.zeros(K)
    for j in range(K):
        if r_pot[j] > 0:
            r_hat[j] = r_obs[j] / r_pot[j]

    # P_verify per edge
    p_verify = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            p_ij = q_hat[i] * r_hat[j]
            p_verify[i, j] = 1.0 - (1.0 - p_ij) ** max(int(N_ij[i, j]), 0)

    validated = [(i, j) for i in range(K) for j in range(K)
                 if i != j and p_verify[i, j] >= p_verify_threshold]

    return {
        "N_ij": N_ij.tolist(),
        "q_hat": q_hat.tolist(),
        "r_hat": r_hat.tolist(),
        "p_verify": p_verify.tolist(),
        "validated_edges": validated,
        "N_min_validated": int(min((N_ij[i, j] for (i, j) in validated), default=0)),
    }


# ----------------------------------------------------------------------
# B7 / B8: per-module OLS respecting candidate / validated graph
# ----------------------------------------------------------------------

def fit_per_module_ols(X_train: np.ndarray, A_train: np.ndarray,
                       parents: Dict[int, List[int]],
                       lag: int = 1) -> Dict[int, np.ndarray]:
    """For each module k with parent module set P_k, fit:
        X[t+1, k] = W_kk @ X[t, k] + B_k @ A[t+1, k]
                  + sum_{i in P_k} W_ik @ tanh(X[t-lag, i]) + bias
    Returns dict mid -> coefficient matrix.
    """
    T, K, n = X_train.shape
    _, _, m = A_train.shape
    coefs: Dict[int, np.ndarray] = {}
    start = max(1, lag + 1)
    for k in range(K):
        P_k = parents.get(k, [])
        # Design matrix per timestep t in [start, T-1]:
        # [ X[t-1, k] (n), A[t, k] (m), tanh(X[t-1-lag, i]) for i in P_k (n each), 1 ]
        feats: List[np.ndarray] = []
        feats.append(X_train[start - 1: T - 1, k])                              # (T-start, n)
        feats.append(A_train[start: T, k])                                       # (T-start, m)
        for i in P_k:
            feats.append(np.tanh(X_train[start - 1 - lag: T - 1 - lag, i]))      # (T-start, n)
        feats.append(np.ones((T - start, 1)))
        Phi = np.concatenate(feats, axis=1)
        Y = X_train[start: T, k]                                                 # (T-start, n)
        W, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
        coefs[k] = W   # (P, n)
    return coefs


def predict_per_module(X_obs: np.ndarray, A_obs: np.ndarray,
                       coefs: Dict[int, np.ndarray],
                       parents: Dict[int, List[int]],
                       lag: int = 1) -> np.ndarray:
    T, K, n = X_obs.shape
    _, _, m = A_obs.shape
    pred = np.zeros_like(X_obs)
    start = max(1, lag + 1)
    pred[:start] = X_obs[:start]
    for k in range(K):
        P_k = parents.get(k, [])
        feats = [X_obs[start - 1: T - 1, k], A_obs[start: T, k]]
        for i in P_k:
            feats.append(np.tanh(X_obs[start - 1 - lag: T - 1 - lag, i]))
        feats.append(np.ones((T - start, 1)))
        Phi = np.concatenate(feats, axis=1)
        pred[start:, k] = Phi @ coefs[k]
    return pred


def observational_candidate_edges(X_train: np.ndarray, K: int,
                                  corr_threshold: float = 0.10) -> List[Tuple[int, int]]:
    """B7: candidate edges from observational data only (no intervention evidence).

    Use lagged cross-correlation: (i -> j) is a candidate if max_q,p corr(X[t-1, i, p], X[t, j, q]) > threshold.
    """
    T = X_train.shape[0]
    n = X_train.shape[2]
    edges = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            best = 0.0
            for p in range(n):
                for q in range(n):
                    a = X_train[:-1, i, p]
                    b = X_train[1:, j, q]
                    if a.std() < 1e-9 or b.std() < 1e-9:
                        continue
                    c = abs(float(np.corrcoef(a, b)[0, 1]))
                    if c > best:
                        best = c
            if best >= corr_threshold:
                edges.append((i, j))
    return edges


# ----------------------------------------------------------------------
# Main run loop
# ----------------------------------------------------------------------

def run_single_seed(seed: int, base_data_dir: str, n_bins: int = 10,
                    eval_split: str = "unseen_int_test") -> Dict[str, Any]:
    data_dir = os.path.join(base_data_dir, f"sanity_chain_d4_seed{seed}")
    obs = load_split(data_dir, "obs")
    eval_data = load_split(data_dir, eval_split)
    int_train = load_split(data_dir, "unseen_int_train")
    oracle = load_oracle(data_dir)
    cfg = oracle["config"]
    K, n, m = cfg["K"], cfg["n_k"], cfg["m_k"]

    X_train = obs["X"]
    A_train = obs["A"]
    X_eval = eval_data["X"]
    A_eval = eval_data["A"]

    # 1. Bin edges from train
    edges = fit_bin_edges(X_train, n_bins=n_bins)
    true_disc = discretize(X_eval, edges)

    # === B10 Oracle ===
    pred_b10 = b10_predict(X_eval, A_eval, oracle, K, n)
    em_b10 = transition_em(discretize(pred_b10, edges), true_disc)

    # === B2 Global Sequence WM ===
    W_global, _ = b2_fit(X_train, A_train)
    pred_b2 = b2_predict(X_eval, A_eval, W_global, K, n, m)
    em_b2 = transition_em(discretize(pred_b2, edges), true_disc)

    # === B7 Causal-no-int ===
    cand_obs = observational_candidate_edges(X_train, K, corr_threshold=0.10)
    parents_b7: Dict[int, List[int]] = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = fit_per_module_ols(X_train, A_train, parents_b7, lag=1)
    pred_b7 = predict_per_module(X_eval, A_eval, coefs_b7, parents_b7, lag=1)
    em_b7 = transition_em(discretize(pred_b7, edges), true_disc)

    # === B8 FedCausalCompose ===
    matcher = synthetic_step3_step4(int_train["X"], int_train["A"],
                                    int_train["intervention_mask"], K, n,
                                    lag_window=3, p_verify_threshold=0.50)
    parents_b8: Dict[int, List[int]] = {k: [] for k in range(K)}
    for (i, j) in matcher["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = fit_per_module_ols(X_train, A_train, parents_b8, lag=1)
    pred_b8 = predict_per_module(X_eval, A_eval, coefs_b8, parents_b8, lag=1)
    em_b8 = transition_em(discretize(pred_b8, edges), true_disc)

    # Theorem 2 trackers
    # Compute the empirical N* such that P_verify >= 0.95 given q_min, r_min
    q_hat = matcher["q_hat"]
    r_hat = matcher["r_hat"]
    # exclude zero rates (modules that never intervened / never received responses)
    q_pos = [q for q in q_hat if q > 0.0]
    r_pos = [r for r in r_hat if r > 0.0]
    q_min = min(q_pos) if q_pos else 0.0
    r_min = min(r_pos) if r_pos else 0.0
    if q_min * r_min > 0:
        # solve 1 - (1 - q_min r_min)^N >= 0.95 -> N >= log(0.05)/log(1 - q_min r_min)
        N_for_p95 = int(np.ceil(np.log(0.05) / np.log(1 - q_min * r_min)))
    else:
        N_for_p95 = float("inf")

    # Ground-truth edges from oracle
    GM_adj = np.asarray(oracle["GM_adj"])
    true_edges = [(int(i), int(j)) for i in range(K) for j in range(K)
                  if GM_adj[i, j] > 0]
    # Edge F1 vs ground truth
    val_set = set((int(i), int(j)) for (i, j) in matcher["validated_edges"])
    true_set = set(true_edges)
    tp = len(val_set & true_set)
    fp = len(val_set - true_set)
    fn = len(true_set - val_set)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    return {
        "seed": seed,
        "K": K, "n_k": n, "m_k": m,
        "true_edges": true_edges,
        "obs_candidate_edges": cand_obs,
        "validated_edges": matcher["validated_edges"],
        "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
        "transition_em": {
            "B2": em_b2, "B7": em_b7, "B8": em_b8, "B10": em_b10,
        },
        "per_module_em": {
            "B2": per_module_em(discretize(pred_b2, edges), true_disc),
            "B7": per_module_em(discretize(pred_b7, edges), true_disc),
            "B8": per_module_em(discretize(pred_b8, edges), true_disc),
            "B10": per_module_em(discretize(pred_b10, edges), true_disc),
        },
        "theorem2": {
            "q_hat": q_hat, "r_hat": r_hat,
            "q_min": q_min, "r_min": r_min,
            "N_min_validated": matcher["N_min_validated"],
            "N_for_p_verify_0_95": N_for_p95,
            "p_verify_diag": [matcher["p_verify"][i][j]
                              for (i, j) in matcher["validated_edges"][:8]],
        },
        "eval_split": eval_split,
        "n_eval_samples": int(np.prod(true_disc.shape)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--out_dir", default="runs/anchor_4")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--eval_split", default="unseen_int_test")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    per_seed = []
    for s in args.seeds:
        print(f"=== seed {s} ===")
        rec = run_single_seed(s, args.data_dir, eval_split=args.eval_split)
        per_seed.append(rec)
        print(f"  EM: B2={rec['transition_em']['B2']:.4f}  "
              f"B7={rec['transition_em']['B7']:.4f}  "
              f"B8={rec['transition_em']['B8']:.4f}  "
              f"B10={rec['transition_em']['B10']:.4f}")
        print(f"  edge_f1={rec['edge_f1']:.3f}  N_min={rec['theorem2']['N_min_validated']}  "
              f"N_for_p95={rec['theorem2']['N_for_p_verify_0_95']}")

    elapsed = time.time() - t0

    # Aggregate across seeds
    def agg(method: str) -> Dict[str, float]:
        vals = [s["transition_em"][method] for s in per_seed]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "values_by_seed": {str(s["seed"]): s["transition_em"][method]
                                   for s in per_seed}}

    summary = {
        "phase": "RUNNING_anchor_4",
        "config": "sanity_chain_d4 (K=5 chain depth=4, alpha=0.8, gamma=0.0, sigma=0.05)",
        "eval_split": args.eval_split,
        "elapsed_seconds": elapsed,
        "per_seed": per_seed,
        "aggregate": {
            "B2_em": agg("B2"),
            "B7_em": agg("B7"),
            "B8_em": agg("B8"),
            "B10_em": agg("B10"),
        },
    }

    # Gate decision
    B2 = summary["aggregate"]["B2_em"]["mean"]
    B7 = summary["aggregate"]["B7_em"]["mean"]
    B8 = summary["aggregate"]["B8_em"]["mean"]
    B10 = summary["aggregate"]["B10_em"]["mean"]
    summary["gate"] = {
        "B8_em_pp": round(B8 * 100, 2),
        "B10_em_pp": round(B10 * 100, 2),
        "B2_em_pp": round(B2 * 100, 2),
        "B7_em_pp": round(B7 * 100, 2),
        "B8_minus_B10_pp": round((B8 - B10) * 100, 2),
        "B8_minus_B2_pp": round((B8 - B2) * 100, 2),
        "criterion_1_pass": (B8 - B10) >= -0.03,
        "criterion_2_pass": (B8 - B2) >= 0.10,
        "N_for_p95_seed0": per_seed[0]["theorem2"]["N_for_p_verify_0_95"],
        "criterion_3_p_verify": bool(per_seed[0]["theorem2"]["N_for_p_verify_0_95"] != float("inf")
                                     and per_seed[0]["theorem2"]["N_for_p_verify_0_95"] < 50),
    }
    summary["gate"]["overall_pass"] = (summary["gate"]["criterion_1_pass"]
                                       and summary["gate"]["criterion_2_pass"])

    out_path = os.path.join(args.out_dir, "anchor4_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(json.dumps(summary["gate"], indent=2))

    return 0 if summary["gate"]["overall_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
