"""
anchor4_v10_horizon.py — rollout horizon scan on V9 SCM (confound_action).

Combines V7 rollout logic with V9 SCM (full Assumption 2). Tests Proposition 1
on intervention-confounded SCM where Theorem 1's δ_int² > 0 actually bites.

Per specification Exp2 G3: Δ(h) monotone, Δ_10 ≥ +7.5pp.

CPU only, no LLM. $0 cost. ~10 min wall.
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
from fed_causal.synthetic_scm_skeleton import simulate                # noqa: E402
from experiments import anchor4_sanity_v2 as v2                       # noqa: E402
from experiments import anchor4_v5_icp as v5                          # noqa: E402
from experiments import anchor4_v6_unseen as v6                       # noqa: E402
from experiments import anchor4_v9_b10fix as v9                       # noqa: E402


def rollout_b10_with_conf(X0, A_seq, U_seq, oracle, K, n, steps):
    """B10 rollout using full Assumption 2 (X[t-1], A[t], U_conf[t])."""
    params = oracle["mechanism_params"]
    cfg = oracle["config"]
    alpha = cfg["alpha"]; gamma = cfg["gamma"]
    GV = oracle["GV_edges"]
    L = max(1, 2 * cfg.get("lag_mean", 1))
    X = np.zeros((steps + L + 1, K, n))
    X[L] = X0
    for t in range(L + 1, L + steps + 1):
        for k in range(K):
            xn = params["W_self"][k] @ X[t - 1, k] + params["B"][k] @ A_seq[t - L - 1, k]
            for (i, p, j, q, lag) in GV:
                if j != k: continue
                xn += alpha * params["W_cross"][(i, j, lag)] @ np.tanh(X[t - 1 - lag, i])
            xn += gamma * params["V"][k] @ U_seq[t - L - 1]
            X[t, k] = xn
    return X[L:]


def rollout_b2(X0, A_seq, W_full, K, n, m, steps):
    """B2 rollout: K-block joint OLS, no U_conf."""
    X = np.zeros((steps + 1, K, n))
    X[0] = X0
    for t in range(steps):
        feat = np.concatenate([X[t].reshape(K * n), A_seq[t].reshape(K * m), [1.0]])
        Xnext_flat = W_full @ feat
        X[t + 1] = Xnext_flat.reshape(K, n)
    return X


def rollout_per_module(X0, A_seq, coefs, parents, K, n, m, steps, lag=1):
    L = max(1, lag + 1)
    X = np.zeros((steps + L, K, n))
    X[L - 1] = X0
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


def b2_fit(X_train, A_train, K, n, m, ridge=1e-3):
    T = X_train.shape[0]
    Xt = X_train[:-1].reshape(T - 1, K * n)
    At = A_train[1:].reshape(T - 1, K * m)
    Y = X_train[1:].reshape(T - 1, K * n)
    Phi = np.concatenate([Xt, At, np.ones((T - 1, 1))], axis=1)
    A_mat = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(A_mat, Phi.T @ Y).T


def eval_rollout(X_eval, A_eval, U_eval, X_train, predict_fn, name, horizons):
    T = X_eval.shape[0]
    max_h = max(horizons)
    rng = np.random.default_rng(0)
    starts = rng.choice(np.arange(0, T - max_h), size=min(200, T - max_h), replace=False)
    K, n = X_eval.shape[1], X_eval.shape[2]
    pred_h = {h: [] for h in horizons}
    true_h = {h: [] for h in horizons}
    for t0 in starts:
        # Predict h-step rollout from X_eval[t0]
        X_traj = predict_fn(X_eval[t0], A_eval[t0:t0 + max_h],
                            U_eval[t0:t0 + max_h] if U_eval is not None else None)
        for h in horizons:
            pred_h[h].append(X_traj[h])
            true_h[h].append(X_eval[t0 + h])

    metrics = {}
    for h in horizons:
        P = np.stack(pred_h[h], axis=0)
        Tn = np.stack(true_h[h], axis=0)
        mse = float(((P - Tn) ** 2).sum(axis=2).mean())
        metrics[f"mse_h{h}"] = mse
        # Bin edges from train
        edges = v2.fit_bin_edges(X_train, n_bins=32)
        Pd = v2.discretize(P, edges)
        Td = v2.discretize(Tn, edges)
        metrics[f"em_h{h}"] = float((Pd == Td).mean())
    return metrics


def run_seed_v10(seed, base_data_dir, config_prefix="confact_chain_d4",
                 alpha_icp=0.05, horizons=(1, 3, 5, 10, 20)):
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

    # === confshift eval (where Theorem 1 bites) ===
    rng_cs = np.random.default_rng(seed * 31 + 23)
    ev_cs = simulate(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                     {"confounder_shift": True, "conf_mu": 2.5}, rng_cs, T=2500)

    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval, U_eval = ev_cs["X"], ev_cs["A"], ev_cs["U_conf"]

    # B2 fit (joint OLS)
    W_b2 = b2_fit(X_train, A_train, K, n, m)
    def pred_b2(X0, A_seq, U=None):
        return rollout_b2(X0, A_seq, W_b2, K, n, m, A_seq.shape[0])
    def pred_b10(X0, A_seq, U=None):
        return rollout_b10_with_conf(X0, A_seq, U, oracle, K, n, A_seq.shape[0])

    # B7 (observational candidate edges) - causal-no-int
    cand_obs = v2.b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = v2.fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)
    def pred_b7(X0, A_seq, U=None):
        return rollout_per_module(X0, A_seq, coefs_b7, parents_b7, K, n, m,
                                  A_seq.shape[0], lag=1)

    # B8d ICP-validated edges
    icp = v5.icp_validate_all(rich["X"], rich["A"], rich["intervention_mask"],
                              K, n, m, alpha=alpha_icp)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in icp["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)
    def pred_b8(X0, A_seq, U=None):
        return rollout_per_module(X0, A_seq, coefs_b8, parents_b8, K, n, m,
                                  A_seq.shape[0], lag=1)

    metrics = {}
    for name, fn, needs_u in [("B10_oracle", pred_b10, True),
                              ("B2_global_seq", pred_b2, False),
                              ("B7_causal_no_int", pred_b7, False),
                              ("B8_fcc", pred_b8, False)]:
        U = U_eval if needs_u else None
        metrics[name] = eval_rollout(X_eval, A_eval, U, X_train, fn, name,
                                     horizons)
    return {"seed": seed, "metrics": metrics,
            "icp_validated_edges": icp["validated_edges"],
            "horizons": list(horizons)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--out_dir", default="runs/anchor_4_v10")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    horizons = (1, 3, 5, 10, 20)

    t0 = time.time()
    per_seed = []
    for s in args.seeds:
        print(f"\n=== seed {s} (confact_chain γ=0.5 γ_A=2.0 + confshift eval) ===", flush=True)
        r = run_seed_v10(s, args.data_dir, horizons=horizons)
        per_seed.append(r)
        # Print table
        print(f"  {'baseline':<22} " + "".join(f"  h={h:>2}_EM h={h:>2}_MSE" for h in horizons))
        for b, m in r["metrics"].items():
            line = f"  {b:<22}"
            for h in horizons:
                em = m[f"em_h{h}"]; mse = m[f"mse_h{h}"]
                line += f"  {em:.3f}   {mse:.2f}  "
            print(line, flush=True)

    elapsed = time.time() - t0
    # Aggregate
    agg_em = {b: {h: 0.0 for h in horizons} for b in ["B10_oracle", "B2_global_seq", "B7_causal_no_int", "B8_fcc"]}
    agg_mse = {b: {h: 0.0 for h in horizons} for b in ["B10_oracle", "B2_global_seq", "B7_causal_no_int", "B8_fcc"]}
    for r in per_seed:
        for b in agg_em:
            for h in horizons:
                agg_em[b][h] += r["metrics"][b][f"em_h{h}"] / len(per_seed)
                agg_mse[b][h] += r["metrics"][b][f"mse_h{h}"] / len(per_seed)

    print()
    print("=" * 100)
    print("AGGREGATE (mean across seeds) — Synthetic SCM with confound_action + confshift eval")
    print("=" * 100)
    print(f"{'Baseline':<22} " + "".join(f"  h={h:>2}_EM  h={h:>2}_MSE" for h in horizons))
    print("-" * 100)
    for b in ["B10_oracle", "B2_global_seq", "B7_causal_no_int", "B8_fcc"]:
        line = f"{b:<22}"
        for h in horizons:
            line += f"  {agg_em[b][h]:.4f}  {agg_mse[b][h]:.3f}  "
        print(line)

    # G3 gate
    print()
    print("=== G3 gate (Δ(h) = B8 EM - B2 EM, monotone & Δ_10 ≥ +7.5pp) ===")
    deltas = {}
    for h in horizons:
        d_em = (agg_em["B8_fcc"][h] - agg_em["B2_global_seq"][h]) * 100
        d_mse = agg_mse["B2_global_seq"][h] - agg_mse["B8_fcc"][h]  # positive = B8 better
        d_mse_b10 = agg_mse["B2_global_seq"][h] - agg_mse["B10_oracle"][h]
        deltas[h] = {"em_pp": round(d_em, 2), "mse_diff_B2-B8": round(d_mse, 3),
                     "mse_diff_B2-B10": round(d_mse_b10, 3)}
        print(f"  h={h:>2}: ΔEM = {d_em:+.2f}pp, MSE B2-B8 = {d_mse:+.3f}, B2-B10 = {d_mse_b10:+.3f}")
    vals_em = [deltas[h]["em_pp"] for h in horizons]
    monotone = all(vals_em[i+1] >= vals_em[i] - 0.01 for i in range(len(vals_em)-1))
    print(f"\n  Monotone (EM): {monotone}")
    print(f"  Δ_10 = {deltas[10]['em_pp']}pp (gate ≥ +7.5pp): {'PASS' if deltas[10]['em_pp'] >= 7.5 else 'FAIL'}")
    # Try also MSE-based (more natural metric for SCM)
    mse_vals = [deltas[h]["mse_diff_B2-B8"] for h in horizons]
    monotone_mse = all(mse_vals[i+1] >= mse_vals[i] - 0.01 for i in range(len(mse_vals)-1))
    print(f"\n  Monotone (MSE B2-B8): {monotone_mse}")
    print(f"  MSE-gap at h=10: B2 MSE - B8 MSE = {deltas[10]['mse_diff_B2-B8']:+.3f}")

    summary = {
        "phase": "exp2_v10_synthetic_rollout",
        "config": "confact_chain_d4 confshift eval",
        "elapsed_seconds": elapsed,
        "horizons": list(horizons),
        "per_seed": per_seed,
        "agg_em": agg_em, "agg_mse": agg_mse,
        "deltas": deltas,
        "G3_em_monotone": monotone,
        "G3_em_delta_10_pp": deltas[10]["em_pp"],
        "G3_em_pass": (monotone and deltas[10]["em_pp"] >= 7.5),
        "G3_mse_monotone": monotone_mse,
    }
    out_path = os.path.join(args.out_dir, "anchor4_v10_horizon_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
