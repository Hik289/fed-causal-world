"""
anchor4_sanity_v2.py — H0.anchor_4 Synthetic Sanity Gate (V2 with fixes).

Diagnostic-driven fixes from V1:

  Fix-1 (MEC resolution): generate a "rich_int" training split that intervenes
        on EACH module 0..K-1 sequentially.  With single-module do() in V1
        we couldn't distinguish 0->1->...->4 chain from 0->{1,2,3,4} star,
        causing edge_f1=0.25 and B8 underfit.

  Fix-2 (B10 EM ceiling): use n_bins=5 instead of 10.  With σ_local=0.05 and
        train std~1.0, 10-bin discretization gives ~25% per-var flip rate from
        noise alone, capping B10 at ~70% — too low for sanity gate.

  Fix-3 (response detection): use lag_window=1 (strict immediate) and require
        the response magnitude to exceed the OBSERVATIONAL-baseline dX
        distribution's 90th percentile (not just median), so the counter is
        not saturated by chain propagation.

  Fix-4 (B8 OLS): use ridge-regularized OLS (ridge=1e-3) to avoid overfit
        when wrong parents are picked, matching the OLS structure used by B7.

This script generates rich_int data in-process (no need to modify the official
synthetic_scm_skeleton.py beyond --force_chain).  It only reads the original
oracle (W_self/B/W_cross/V/k_mediator) and re-simulates trajectories with rich
interventions.
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

# Bring in the official generator's simulate() so we use the exact dynamics
sys.path.insert(0, "/home/user/fedcausalworld/data")
from synthetic_scm_skeleton import SCMConfig, simulate  # noqa: E402


# ----------------------------------------------------------------------
# Load oracle + obs + re-simulate rich_int and rich_eval splits
# ----------------------------------------------------------------------

def load_npz(path):
    return dict(np.load(path, allow_pickle=False))


def reload_cfg(oracle_cfg: Dict[str, Any]) -> SCMConfig:
    """Reconstruct SCMConfig from oracle.pkl's config dict."""
    return SCMConfig(**{k: v for k, v in oracle_cfg.items()
                        if k in SCMConfig.__dataclass_fields__})


def gen_rich_int_train(cfg: SCMConfig, GV_edges, params, rng,
                       T_per_module: int = 800) -> Dict[str, np.ndarray]:
    """Generate train trajectories with do(A_k) on each module k separately.

    Concatenated into one stream so the downstream matcher sees a wide variety
    of intervention sources, resolving the K-1 interventions needed by
    Lemma 3.
    """
    parts = []
    masks = []
    for k_do in range(cfg.K):
        spec = {"do_module": k_do,
                "do_value": np.ones(cfg.m_k),
                "rate": 0.30}
        out = simulate(cfg, GV_edges, params, spec, rng, T_per_module)
        parts.append(out)
        masks.append(out["intervention_mask"])
    X = np.concatenate([p["X"] for p in parts], axis=0)
    A = np.concatenate([p["A"] for p in parts], axis=0)
    M = np.concatenate(masks, axis=0)
    return {"X": X, "A": A, "intervention_mask": M}


def gen_mediator_eval(cfg: SCMConfig, GV_edges, params, rng, T: int = 1500
                      ) -> Dict[str, np.ndarray]:
    """Eval split: intervene on chain-MIDDLE module so the effect propagates
    forward and creates a genuine joint-distribution shift the non-causal
    global OLS cannot handle by extrapolation alone."""
    spec = {"do_module": cfg.K // 2,
            "do_value": np.ones(cfg.m_k),
            "rate": 0.40}
    return simulate(cfg, GV_edges, params, spec, rng, T)


# ----------------------------------------------------------------------
# Discretization (coarser: 5 bins)
# ----------------------------------------------------------------------

def fit_bin_edges(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    T, K, n = X.shape
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.empty((K, n, n_bins + 1))
    for k in range(K):
        for q in range(n):
            edges[k, q] = np.quantile(X[:, k, q], qs)
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


def transition_em(pred_d, true_d):
    return float((pred_d == true_d).mean())


def per_module_em(pred_d, true_d):
    return {int(k): float((pred_d[:, k] == true_d[:, k]).mean())
            for k in range(pred_d.shape[1])}


# ----------------------------------------------------------------------
# B10: Oracle Causal WM (uses ground-truth params)
# ----------------------------------------------------------------------

def b10_predict(X, A, oracle, K, n):
    params = oracle["mechanism_params"]
    cfg = oracle["config"]
    alpha = cfg["alpha"]
    GV_edges = oracle["GV_edges"]
    T = X.shape[0]
    pred = np.zeros_like(X)
    pred[0] = X[0]
    for t in range(1, T):
        for k in range(K):
            xn = params["W_self"][k] @ X[t - 1, k] + params["B"][k] @ A[t, k]
            for (i, p, j, q, lag) in GV_edges:
                if j != k:
                    continue
                t_src = max(0, t - 1 - lag)
                xn += alpha * params["W_cross"][(i, j, lag)] @ np.tanh(X[t_src, i])
            pred[t, k] = xn
    return pred


# ----------------------------------------------------------------------
# B2: Global Sequence WM (joint OLS, no graph)
# ----------------------------------------------------------------------

def b2_fit_predict(X_train, A_train, X_eval, A_eval, K, n, m, ridge: float = 1e-3):
    T = X_train.shape[0]
    Xt = X_train[:-1].reshape(T - 1, K * n)
    At = A_train[1:].reshape(T - 1, K * m)
    Y = X_train[1:].reshape(T - 1, K * n)
    Phi = np.concatenate([Xt, At, np.ones((T - 1, 1))], axis=1)
    # ridge regression
    A_mat = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    B_mat = Phi.T @ Y
    W = np.linalg.solve(A_mat, B_mat).T  # (K*n, P)
    Te = X_eval.shape[0]
    Xe = X_eval[:-1].reshape(Te - 1, K * n)
    Ae = A_eval[1:].reshape(Te - 1, K * m)
    Phi_e = np.concatenate([Xe, Ae, np.ones((Te - 1, 1))], axis=1)
    pred = np.zeros_like(X_eval)
    pred[1:] = (Phi_e @ W.T).reshape(Te - 1, K, n)
    pred[0] = X_eval[0]
    return pred


# ----------------------------------------------------------------------
# Step 3+4 (numerical): rich-intervention edge recovery
# ----------------------------------------------------------------------

def rich_step3_step4(X_train, A_train, int_mask, X_obs_baseline, K, n,
                     lag_window: int = 1, p_verify_threshold: float = 0.50
                     ) -> Dict[str, Any]:
    """Strict-lag response matching with calibrated threshold from non-int baseline.

    Procedure:
      - q_i := mean(int_mask[:, i])  (rich split: should be ~0.3 / K each)
      - response detection per (t, k): dX[t, k] = ||X[t] - X[t-1]|| at module k
      - threshold = 90th percentile of dX[no_intervention_anywhere_t, k]
                    (per-module calibration from OBSERVATIONAL baseline)
      - For each t where module i has do(): look at t+1 only; if dX[t+1, j]
        > thresh[j], count N_ij[i, j] +=1.
    """
    T = X_train.shape[0]
    dX = np.linalg.norm(X_train[1:] - X_train[:-1], axis=2)            # (T-1, K)
    # Calibrate threshold from obs baseline (no interventions)
    dX_obs = np.linalg.norm(X_obs_baseline[1:] - X_obs_baseline[:-1], axis=2)  # (T_obs-1, K)
    thresh = np.quantile(dX_obs, 0.90, axis=0)                          # (K,)

    q_hat = np.array([int_mask[:, i].mean() for i in range(K)])

    N_ij = np.zeros((K, K), dtype=int)
    r_pot = np.zeros(K, dtype=int)
    r_obs = np.zeros(K, dtype=int)
    for t in range(T - 1):
        for i in range(K):
            if int_mask[t, i] == 0:
                continue
            # strict lag = 1: look only at t+1 - t in dX is index t -> use dX[t]
            for j in range(K):
                if i == j:
                    continue
                r_pot[j] += 1
                if dX[t, j] > thresh[j]:
                    r_obs[j] += 1
                    N_ij[i, j] += 1

    r_hat = np.zeros(K)
    for j in range(K):
        if r_pot[j] > 0:
            r_hat[j] = r_obs[j] / r_pot[j]

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
        "thresh": thresh.tolist(),
        "validated_edges": validated,
        "N_min_validated": int(min((N_ij[i, j] for (i, j) in validated), default=0)),
    }


# ----------------------------------------------------------------------
# Per-module ridge OLS for B7/B8
# ----------------------------------------------------------------------

def fit_per_module_ridge(X_train, A_train, parents: Dict[int, List[int]],
                         lag: int = 1, ridge: float = 1e-3) -> Dict[int, np.ndarray]:
    T, K, n = X_train.shape
    _, _, m = A_train.shape
    out = {}
    start = max(1, lag + 1)
    for k in range(K):
        P_k = parents.get(k, [])
        feats = [X_train[start - 1: T - 1, k],
                 A_train[start: T, k]]
        for i in P_k:
            feats.append(np.tanh(X_train[start - 1 - lag: T - 1 - lag, i]))
        feats.append(np.ones((T - start, 1)))
        Phi = np.concatenate(feats, axis=1)
        Y = X_train[start: T, k]
        A_mat = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
        B_mat = Phi.T @ Y
        out[k] = np.linalg.solve(A_mat, B_mat)
    return out


def predict_per_module(X, A, coefs, parents, lag=1):
    T, K, n = X.shape
    _, _, m = A.shape
    pred = np.zeros_like(X)
    start = max(1, lag + 1)
    pred[:start] = X[:start]
    for k in range(K):
        P_k = parents.get(k, [])
        feats = [X[start - 1: T - 1, k],
                 A[start: T, k]]
        for i in P_k:
            feats.append(np.tanh(X[start - 1 - lag: T - 1 - lag, i]))
        feats.append(np.ones((T - start, 1)))
        Phi = np.concatenate(feats, axis=1)
        pred[start:, k] = Phi @ coefs[k]
    return pred


def b7_candidate_edges(X_train, K, threshold=0.10) -> List[Tuple[int, int]]:
    """B7 observational candidate: max lagged cross-correlation > threshold."""
    edges = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            best = 0.0
            for p in range(X_train.shape[2]):
                for q in range(X_train.shape[2]):
                    a = X_train[:-1, i, p]
                    b = X_train[1:, j, q]
                    if a.std() < 1e-9 or b.std() < 1e-9:
                        continue
                    c = abs(float(np.corrcoef(a, b)[0, 1]))
                    if c > best:
                        best = c
            if best >= threshold:
                edges.append((i, j))
    return edges


# ----------------------------------------------------------------------
# Run single seed
# ----------------------------------------------------------------------

def run_seed(seed: int, base_data_dir: str, n_bins: int = 5) -> Dict[str, Any]:
    data_dir = os.path.join(base_data_dir, f"sanity_chain_d4_seed{seed}")
    obs = load_npz(os.path.join(data_dir, "obs.npz"))
    with open(os.path.join(data_dir, "oracle.pkl"), "rb") as f:
        oracle = pickle.load(f)
    cfg = reload_cfg(oracle["config"])
    K, n, m = cfg.K, cfg.n_k, cfg.m_k

    # === Generate rich-int train + mediator-int eval splits ===
    rng = np.random.default_rng(seed * 31 + 17)
    rich = gen_rich_int_train(cfg, oracle["GV_edges"],
                              oracle["mechanism_params"], rng, T_per_module=800)
    rng2 = np.random.default_rng(seed * 31 + 19)
    eval_split = gen_mediator_eval(cfg, oracle["GV_edges"],
                                   oracle["mechanism_params"], rng2, T=1500)

    X_train = obs["X"]
    A_train = obs["A"]
    X_eval = eval_split["X"]
    A_eval = eval_split["A"]

    # Bin edges fit on train obs (B2/B7/B8 trained on this).
    edges = fit_bin_edges(X_train, n_bins=n_bins)
    true_disc = discretize(X_eval, edges)

    # --- B10 Oracle ---
    pred_b10 = b10_predict(X_eval, A_eval, oracle, K, n)
    em_b10 = transition_em(discretize(pred_b10, edges), true_disc)

    # --- B2 Global Sequence WM ---
    pred_b2 = b2_fit_predict(X_train, A_train, X_eval, A_eval, K, n, m)
    em_b2 = transition_em(discretize(pred_b2, edges), true_disc)

    # --- B7 (observational candidate edges) ---
    cand_obs = b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)
    pred_b7 = predict_per_module(X_eval, A_eval, coefs_b7, parents_b7, lag=1)
    em_b7 = transition_em(discretize(pred_b7, edges), true_disc)

    # --- B8 FedCausalCompose using rich_int + lag=1 strict ---
    matcher = rich_step3_step4(rich["X"], rich["A"],
                               rich["intervention_mask"], X_train, K, n,
                               lag_window=1, p_verify_threshold=0.50)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in matcher["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)
    pred_b8 = predict_per_module(X_eval, A_eval, coefs_b8, parents_b8, lag=1)
    em_b8 = transition_em(discretize(pred_b8, edges), true_disc)

    # Theorem 2 trackers
    q_pos = [q for q in matcher["q_hat"] if q > 0]
    r_pos = [r for r in matcher["r_hat"] if r > 0]
    q_min = min(q_pos) if q_pos else 0.0
    r_min = min(r_pos) if r_pos else 0.0
    if q_min * r_min > 0:
        N_for_p95 = int(np.ceil(np.log(0.05) / np.log(1 - q_min * r_min)))
    else:
        N_for_p95 = float("inf")

    GM_adj = np.asarray(oracle["GM_adj"])
    true_edges = [(int(i), int(j)) for i in range(K) for j in range(K)
                  if GM_adj[i, j] > 0]
    val_set = set((int(i), int(j)) for (i, j) in matcher["validated_edges"])
    true_set = set(true_edges)
    tp = len(val_set & true_set)
    fp = len(val_set - true_set)
    fn = len(true_set - val_set)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    return {
        "seed": seed, "K": K, "n_k": n, "m_k": m,
        "true_edges": true_edges,
        "obs_candidate_edges": cand_obs,
        "validated_edges": matcher["validated_edges"],
        "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
        "transition_em": {"B2": em_b2, "B7": em_b7, "B8": em_b8, "B10": em_b10},
        "per_module_em": {
            "B2": per_module_em(discretize(pred_b2, edges), true_disc),
            "B7": per_module_em(discretize(pred_b7, edges), true_disc),
            "B8": per_module_em(discretize(pred_b8, edges), true_disc),
            "B10": per_module_em(discretize(pred_b10, edges), true_disc),
        },
        "theorem2": {
            "q_hat": matcher["q_hat"], "r_hat": matcher["r_hat"],
            "q_min": q_min, "r_min": r_min,
            "N_min_validated": matcher["N_min_validated"],
            "N_for_p_verify_0_95": N_for_p95,
        },
        "eval_split": "mediator_int_resimulated",
        "n_eval_samples": int(np.prod(true_disc.shape)),
        "n_bins": n_bins,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/user/fedcausalworld/data/synthetic")
    ap.add_argument("--out_dir", default="/home/user/fedcausalworld/experiments/anchor_4_run")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n_bins", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    per_seed = []
    for s in args.seeds:
        print(f"=== seed {s} ===")
        rec = run_seed(s, args.data_dir, n_bins=args.n_bins)
        per_seed.append(rec)
        em = rec["transition_em"]
        print(f"  EM: B2={em['B2']:.4f}  B7={em['B7']:.4f}  "
              f"B8={em['B8']:.4f}  B10={em['B10']:.4f}")
        print(f"  edge_f1={rec['edge_f1']:.3f}  validated_edges={rec['validated_edges']}")
        print(f"  N_min={rec['theorem2']['N_min_validated']}  "
              f"N_for_p95={rec['theorem2']['N_for_p_verify_0_95']}")

    elapsed = time.time() - t0

    def agg(method):
        vals = [s["transition_em"][method] for s in per_seed]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "values_by_seed": {str(s["seed"]): s["transition_em"][method]
                                   for s in per_seed}}

    B2 = agg("B2"); B7 = agg("B7"); B8 = agg("B8"); B10 = agg("B10")
    gate = {
        "B10_em_pp": round(B10["mean"] * 100, 2),
        "B8_em_pp": round(B8["mean"] * 100, 2),
        "B7_em_pp": round(B7["mean"] * 100, 2),
        "B2_em_pp": round(B2["mean"] * 100, 2),
        "B8_minus_B10_pp": round((B8["mean"] - B10["mean"]) * 100, 2),
        "B8_minus_B2_pp": round((B8["mean"] - B2["mean"]) * 100, 2),
        "criterion_1_pass": (B8["mean"] - B10["mean"]) >= -0.03,
        "criterion_2_pass": (B8["mean"] - B2["mean"]) >= 0.10,
        "N_for_p95_seed0": per_seed[0]["theorem2"]["N_for_p_verify_0_95"],
        "criterion_3_p_verify": (per_seed[0]["theorem2"]["N_for_p_verify_0_95"] != float("inf")
                                  and per_seed[0]["theorem2"]["N_for_p_verify_0_95"] < 50),
    }
    gate["overall_pass"] = gate["criterion_1_pass"] and gate["criterion_2_pass"]

    summary = {
        "phase": "RUNNING_anchor_4_v2",
        "config": "sanity_chain_d4 (K=5, chain depth=4, alpha=0.8, gamma=0.0, sigma=0.05)",
        "version_notes": [
            "V2: rich_int train (cycle do() over all K modules) — Lemma 3 K interventions",
            "V2: n_bins=5 (V1 used 10; B10 EM was floor-capped by sigma/bin_width)",
            "V2: lag_window=1 strict + 90th-percentile dX threshold from obs baseline",
            "V2: eval = mediator-intervention (chain MIDDLE module), forward-propagating",
            "V2: B7 + B8 use ridge OLS (ridge=1e-3)",
        ],
        "elapsed_seconds": elapsed,
        "per_seed": per_seed,
        "aggregate": {"B2_em": B2, "B7_em": B7, "B8_em": B8, "B10_em": B10},
        "gate": gate,
    }
    out_path = os.path.join(args.out_dir, "anchor4_results_v2.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}")
    print(json.dumps(gate, indent=2))
    return 0 if gate["overall_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
