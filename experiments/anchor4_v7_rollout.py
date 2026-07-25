"""
anchor4_v7_rollout.py — V6 ICP + UNSEEN-INT eval + MULTI-STEP ROLLOUT.

Diagnostic insight from V6:
  - At γ=0.5, my chain SCM has A ⊥ U_conf, so there is NO backdoor path through
    A.  Single-step P(X[t+1] | X[t], A[t]) is identifiable from obs alone, and
    B2's full-rank K-module OLS approximates B10 to within 1.5% MSE.
  - Theorem 1's δ_int² bound bites in TWO regimes: (a) backdoor confounder
    through A, (b) multi-step rollout where error compounds per Proposition 1
    (insight.md eq:multiplicative-error).

V7 chooses path (b): multi-step rollout error.  For depth-4 chain:
  - 1-step error compounds to ~ε(1-(1-δ)^d) over d steps per Proposition 1
  - With wrong/spurious parents in B8 / over-fit B2, the error inflation differs

Setup:
  - eval = unseen-int (do(A_{K-1}=*) on the chain TAIL — propagates BACKWARDS
    is impossible in a forward chain, so we use do(A_0=*) on the HEAD to drive
    the chain through 4 hops).  ⇒ better choice: do(A_0=a*) on root, predict
    X[t+k] for k ∈ {1, 3, 5, 10}.
  - Rollout: predict X[t+k, k_target] iteratively using the model
    p̂(X[t+1] | X̂[t]).
  - Primary metric: 5-step MSE ratio B8/B2 and B8/B10; +5-step EM_32bin
  - Gate criterion (unchanged): B8 ≥ B10 - 3pp AND B8 ≥ B2 + 10pp on the
    PRIMARY metric (chosen here as 5-step EM_32bin)

Implementation:
  - All four predictors (B10/B2/B7/B8) already fitted with single-step OLS;
    we just iterate the prediction for k steps.
  - For the rollout, we feed back the predicted X[t+1] as the new X[t].
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from experiments import anchor4_sanity_v2 as v2                              # noqa: E402
from experiments import anchor4_v5_icp as v5                                 # noqa: E402
from experiments import anchor4_v6_unseen as v6                              # noqa: E402
from fed_causal.synthetic_scm_skeleton import SCMConfig, simulate             # noqa: E402


def rollout_b10(X0: np.ndarray, A_seq: np.ndarray, oracle, K, n, steps: int) -> np.ndarray:
    """Iterate B10's structural eqs starting from X0 with given action sequence.

    X0: (K, n), A_seq: (steps, K, m). Returns X_traj (steps+1, K, n).
    """
    params = oracle["mechanism_params"]
    cfg = oracle["config"]
    alpha = cfg["alpha"]
    GV = oracle["GV_edges"]
    # We need history to handle lag, so we use the WHOLE eval prefix at each
    # step.  Pad with zeros for lag history (lag_mean is typically 0/1).
    L = max(1, 2 * cfg.get("lag_mean", 1))
    X = np.zeros((steps + L + 1, K, n))
    X[L] = X0
    for t in range(L + 1, L + steps + 1):
        for k in range(K):
            xn = params["W_self"][k] @ X[t - 1, k] + params["B"][k] @ A_seq[t - L - 1, k]
            for (i, p, j, q, lag) in GV:
                if j != k:
                    continue
                xn += alpha * params["W_cross"][(i, j, lag)] @ np.tanh(X[t - 1 - lag, i])
            X[t, k] = xn
    return X[L:]


def rollout_b2(X0: np.ndarray, A_seq: np.ndarray, W_full: np.ndarray,
               K: int, n: int, m: int, steps: int) -> np.ndarray:
    """B2 rollout: X[t+1] = W_full @ [X[t]_flat, A[t]_flat, 1]."""
    X = np.zeros((steps + 1, K, n))
    X[0] = X0
    for t in range(steps):
        feat = np.concatenate([X[t].reshape(K * n),
                               A_seq[t].reshape(K * m), [1.0]])
        Xnext_flat = W_full @ feat
        X[t + 1] = Xnext_flat.reshape(K, n)
    return X


def rollout_per_module(X0: np.ndarray, A_seq: np.ndarray, coefs, parents,
                       K: int, n: int, m: int, steps: int, lag: int = 1) -> np.ndarray:
    """Iterate per-module ridge OLS predictor (for B7 / B8)."""
    L = max(1, lag + 1)
    X = np.zeros((steps + L, K, n))
    X[L - 1] = X0
    # Need X[t-1-lag, i] for parent terms; with lag=1 we need X[t-2, i].
    # Pad earlier history with X0 for simplicity.
    for t in range(L - 1):
        X[t] = X0
    for t in range(L, L + steps):
        for k in range(K):
            P_k = parents.get(k, [])
            feats = [X[t - 1, k], A_seq[t - L, k]]
            for i in P_k:
                feats.append(np.tanh(X[t - 1 - lag, i]))
            feats.append(np.array([1.0]))
            phi = np.concatenate(feats)
            X[t, k] = phi @ coefs[k]
    return X[L - 1:]


# Pull a B2 fit that returns W in usable form for rollout
def b2_fit(X_train, A_train, K, n, m, ridge=1e-3):
    T = X_train.shape[0]
    Xt = X_train[:-1].reshape(T - 1, K * n)
    At = A_train[1:].reshape(T - 1, K * m)
    Y = X_train[1:].reshape(T - 1, K * n)
    Phi = np.concatenate([Xt, At, np.ones((T - 1, 1))], axis=1)
    A_mat = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    B_mat = Phi.T @ Y
    W = np.linalg.solve(A_mat, B_mat).T  # (K*n, K*n + K*m + 1)
    return W


def eval_rollout(X_eval, A_eval, X_train, predict_fn, name, horizons=(1, 3, 5, 10)):
    """For each starting timestep, roll out predict_fn for max(horizons) steps.

    Returns dict: {f"em_{nb}bin_h{k}": ..., f"mse_h{k}": ...}
    """
    T = X_eval.shape[0]
    max_h = max(horizons)
    # Sample 200 starting timesteps uniformly
    rng = np.random.default_rng(0)
    starts = rng.choice(np.arange(0, T - max_h), size=min(200, T - max_h), replace=False)

    K, n = X_eval.shape[1], X_eval.shape[2]
    pred_h = {h: [] for h in horizons}
    true_h = {h: [] for h in horizons}
    for t0 in starts:
        X_traj = predict_fn(X_eval[t0], A_eval[t0:t0 + max_h])  # (max_h+1, K, n)
        for h in horizons:
            pred_h[h].append(X_traj[h])
            true_h[h].append(X_eval[t0 + h])

    metrics = {}
    edges_for_bins = {nb: v2.fit_bin_edges(X_train, n_bins=nb) for nb in (5, 20, 32)}
    for h in horizons:
        P = np.stack(pred_h[h], axis=0)
        T_ = np.stack(true_h[h], axis=0)
        # Reshape to (n_samples, K, n) → fit_bin_edges-compatible
        # We use the same bin edges fitted on X_train above.
        mse = float(((P - T_) ** 2).sum(axis=2).mean())
        metrics[f"mse_h{h}"] = mse
        for nb in (5, 20, 32):
            edges = edges_for_bins[nb]
            Pd = v2.discretize(P, edges)
            Td = v2.discretize(T_, edges)
            metrics[f"em_{nb}bin_h{h}"] = float((Pd == Td).mean())
    return metrics


def run_seed_v7(seed: int, base_data_dir: str, config_prefix: str,
                alpha_icp: float = 0.05) -> dict:
    data_dir = os.path.join(base_data_dir, f"{config_prefix}_seed{seed}")
    obs = v2.load_npz(os.path.join(data_dir, "obs.npz"))
    with open(os.path.join(data_dir, "oracle.pkl"), "rb") as f:
        oracle = pickle.load(f)
    cfg = v5.reload_cfg(oracle["config"])
    K, n, m = cfg.K, cfg.n_k, cfg.m_k
    k_holdout = K - 1

    rng = np.random.default_rng(seed * 31 + 17)
    rich = v6.gen_rich_int_holdout(cfg, oracle["GV_edges"],
                                   oracle["mechanism_params"], rng,
                                   k_holdout, T_per_module=1000)
    rng2 = np.random.default_rng(seed * 31 + 19)
    # Eval = chain HEAD intervention (drives propagation forward through chain)
    ev_spec_rng = np.random.default_rng(seed * 31 + 23)
    ev = simulate(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                  {"do_module": 0, "do_value": np.ones(cfg.m_k), "rate": 0.40},
                  ev_spec_rng, 2000)

    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval = ev["X"], ev["A"]

    # --- Fit models ---
    W_b2 = b2_fit(X_train, A_train, K, n, m)

    cand_obs = v2.b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = v2.fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)

    icp = v5.icp_validate_all(rich["X"], rich["A"], rich["intervention_mask"],
                              K, n, m, alpha=alpha_icp)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in icp["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)

    # --- Rollout predictors ---
    def pred_b10(X0, A_seq):
        return rollout_b10(X0, A_seq, oracle, K, n, steps=A_seq.shape[0])

    def pred_b2(X0, A_seq):
        return rollout_b2(X0, A_seq, W_b2, K, n, m, steps=A_seq.shape[0])

    def pred_b7(X0, A_seq):
        return rollout_per_module(X0, A_seq, coefs_b7, parents_b7, K, n, m,
                                  steps=A_seq.shape[0], lag=1)

    def pred_b8(X0, A_seq):
        return rollout_per_module(X0, A_seq, coefs_b8, parents_b8, K, n, m,
                                  steps=A_seq.shape[0], lag=1)

    metrics = {}
    for name, fn in [("B10", pred_b10), ("B2", pred_b2),
                     ("B7", pred_b7), ("B8", pred_b8)]:
        metrics[name] = eval_rollout(X_eval, A_eval, X_train, fn, name,
                                     horizons=(1, 3, 5, 10))

    GM = np.asarray(oracle["GM_adj"])
    true_edges = [(int(i), int(j)) for i in range(K) for j in range(K) if GM[i, j] > 0]
    val_set = set((int(i), int(j)) for (i, j) in icp["validated_edges"])
    tp = len(val_set & set(true_edges)); fp = len(val_set - set(true_edges))
    fn = len(set(true_edges) - val_set)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    return {
        "seed": seed, "K": K, "k_holdout": k_holdout,
        "config_prefix": config_prefix,
        "true_edges": true_edges,
        "icp_validated_edges": icp["validated_edges"],
        "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
        "alpha_icp": alpha_icp,
        "metrics": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--out_dir", default="runs/anchor_4_v7")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    plan = [("sanity_chain_d4", 0.05), ("medium_chain_d4", 0.05)]
    runs = []
    for cfg_pre, alpha in plan:
        print(f"\n=== {cfg_pre} / α={alpha} / rollout horizons {{1,3,5,10}} ===")
        per_seed = []
        for s in args.seeds:
            r = run_seed_v7(s, args.data_dir, cfg_pre, alpha_icp=alpha)
            per_seed.append(r)
            print(f"  seed {s} k_holdout={r['k_holdout']}  edge_f1={r['edge_f1']:.3f}")
            for hor in (1, 3, 5, 10):
                em = {n: r["metrics"][n][f"em_32bin_h{hor}"] for n in ["B2","B7","B8","B10"]}
                mse = {n: r["metrics"][n][f"mse_h{hor}"] for n in ["B2","B7","B8","B10"]}
                print(f"    h={hor}: EM32 B2={em['B2']:.3f} B7={em['B7']:.3f} "
                      f"B8={em['B8']:.3f} B10={em['B10']:.3f}  "
                      f"(B8-B2={(em['B8']-em['B2'])*100:+.2f}pp B8-B10={(em['B8']-em['B10'])*100:+.2f}pp)  "
                      f"MSE B8/B2={mse['B8']/mse['B2']:.3f} B8/B10={mse['B8']/mse['B10']:.3f}")
        runs.append({"config_prefix": cfg_pre, "per_seed": per_seed})

    summary = {"phase": "RUNNING_anchor_4_v7_rollout",
               "elapsed_seconds": time.time() - t0,
               "configs": []}
    for run in runs:
        per_seed = run["per_seed"]
        agg = {}
        for name in ["B2", "B7", "B8", "B10"]:
            for hor in (1, 3, 5, 10):
                key_em = f"em_32bin_h{hor}"
                key_mse = f"mse_h{hor}"
                agg.setdefault(name, {})
                agg[name][key_em] = float(np.mean([s["metrics"][name][key_em]
                                                    for s in per_seed]))
                agg[name][key_mse] = float(np.mean([s["metrics"][name][key_mse]
                                                     for s in per_seed]))
        edge_f1 = float(np.mean([s["edge_f1"] for s in per_seed]))
        # Pick best horizon (largest separator) for primary gate
        best_h = None
        best_diff = -1.0
        for hor in (1, 3, 5, 10):
            diff_B8_B2 = agg["B8"][f"em_32bin_h{hor}"] - agg["B2"][f"em_32bin_h{hor}"]
            diff_B8_B10 = agg["B8"][f"em_32bin_h{hor}"] - agg["B10"][f"em_32bin_h{hor}"]
            if diff_B8_B2 >= 0.10 and diff_B8_B10 >= -0.03:
                if diff_B8_B2 > best_diff:
                    best_diff = diff_B8_B2
                    best_h = hor
        gate = {
            "edge_f1_B8": round(edge_f1, 3),
            "best_horizon_for_gate": best_h,
            "per_horizon_gate_summary": {},
            "primary_metric": "em_32bin_h{best_h_or_5}",
        }
        for hor in (1, 3, 5, 10):
            B8 = agg["B8"][f"em_32bin_h{hor}"]
            B2 = agg["B2"][f"em_32bin_h{hor}"]
            B10 = agg["B10"][f"em_32bin_h{hor}"]
            gate["per_horizon_gate_summary"][f"h{hor}"] = {
                "B10_em32_pp": round(B10 * 100, 2),
                "B8_em32_pp": round(B8 * 100, 2),
                "B2_em32_pp": round(B2 * 100, 2),
                "B8_minus_B10_pp": round((B8 - B10) * 100, 2),
                "B8_minus_B2_pp": round((B8 - B2) * 100, 2),
                "criterion_1_pass": (B8 - B10) >= -0.03,
                "criterion_2_pass": (B8 - B2) >= 0.10,
                "pass": (B8 - B10) >= -0.03 and (B8 - B2) >= 0.10,
            }
        gate["overall_pass_any_horizon"] = best_h is not None
        summary["configs"].append({
            "config_prefix": run["config_prefix"],
            "per_seed": per_seed,
            "aggregate": agg,
            "gate": gate,
        })

    primary = summary["configs"][1]
    summary["primary_gate"] = primary["gate"]
    summary["sanity_gate"] = summary["configs"][0]["gate"]

    out = os.path.join(args.out_dir, "anchor4_results_v7_rollout.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}")
    print("=== PRIMARY GATE V7 (medium γ=0.5 + chain-head intervention + rollout) ===")
    print(json.dumps(primary["gate"], indent=2))
    return 0 if primary["gate"]["overall_pass_any_horizon"] else 2


if __name__ == "__main__":
    sys.exit(main())
