"""
exp7_g6_g7.py — Synthetic SCM analysis for G6 + G7

G7 (intervention coverage scaling):
  For each p_int level ∈ {0, 0.05, 0.10, 0.25, 0.50, 1.0}, subsample the
  rich-int training data to keep only fraction p_int of intervention events.
  Run ICP F-test on subsampled data and measure:
    - edge_f1 (graph recovery quality)
    - B8d MSE on confshift eval (downstream prediction)
    - N_min (count of matched intervention events)
  Compare with Theorem 2 prediction: N_min ≈ log(|E_M*|/δ)/(q_min·r_min) ≈ 12-51
  Saturation: marginal gain ≤ 1pp after p_int ≥ 0.25.

G6 (Spearman ρ of Δ_causal vs depth d / α / γ):
  Re-run anchor_4 v9-style pipeline at varied (d, α, γ) parameter sweeps.
  Δ_causal = B2 MSE - B10 MSE (oracle vs non-causal) at h=10.
  Compute Spearman ρ of Δ_causal vs each parameter.
  Predict: ρ ≥ 0.7 for d (Proposition 1) and γ (Theorem 1).

CPU only, ~30-60 min.
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from experiments import anchor4_sanity_v2 as v2                              # noqa: E402
from experiments import anchor4_v5_icp as v5                                 # noqa: E402
from experiments import anchor4_v6_unseen as v6                              # noqa: E402
from fed_causal.synthetic_scm_skeleton import SCMConfig, simulate            # noqa: E402


# =========================================================
# G7: Intervention coverage scaling
# =========================================================

def subsample_rich(rich: dict, p_int: float, seed: int = 0) -> dict:
    """Keep only fraction p_int of intervention events; zero out the rest."""
    rng = np.random.default_rng(seed)
    int_mask = rich["intervention_mask"].copy()
    T, K = int_mask.shape
    # find all (t, k) where intervention occurred
    int_locs = np.argwhere(int_mask == 1)
    n_int = len(int_locs)
    keep_n = int(np.round(p_int * n_int))
    if keep_n < n_int:
        keep_idx = rng.choice(n_int, size=keep_n, replace=False)
        new_mask = np.zeros_like(int_mask)
        for idx in keep_idx:
            t, k = int_locs[idx]
            new_mask[t, k] = 1
        rich = dict(rich)
        rich["intervention_mask"] = new_mask
    return rich


def g7_run_seed(seed: int, base_data_dir: str, p_ints: List[float],
                config_prefix: str = "confact_chain_d4") -> Dict[str, Any]:
    data_dir = os.path.join(base_data_dir, f"{config_prefix}_seed{seed}")
    obs = v2.load_npz(os.path.join(data_dir, "obs.npz"))
    with open(os.path.join(data_dir, "oracle.pkl"), "rb") as f:
        oracle = pickle.load(f)
    cfg = v5.reload_cfg(oracle["config"])
    K, n, m = cfg.K, cfg.n_k, cfg.m_k
    k_holdout = K - 1

    rng = np.random.default_rng(seed * 31 + 17)
    rich_full = v6.gen_rich_int_holdout(cfg, oracle["GV_edges"],
                                        oracle["mechanism_params"], rng,
                                        k_holdout, T_per_module=1000)
    rng_cs = np.random.default_rng(seed * 31 + 23)
    ev_cs = simulate(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                     {"confounder_shift": True, "conf_mu": 2.5}, rng_cs, T=1500)

    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval = ev_cs["X"], ev_cs["A"]

    GM_adj = np.asarray(oracle["GM_adj"])
    true_edges = [(int(i), int(j)) for i in range(K) for j in range(K) if GM_adj[i, j] > 0]

    # Reference: oracle MSE
    pred_b10 = v2.b10_predict(X_eval, A_eval, oracle, K, n)
    b10_mse = v5.cont_mse(pred_b10, X_eval)
    # B2 baseline (always same regardless of p_int)
    pred_b2 = v2.b2_fit_predict(X_train, A_train, X_eval, A_eval, K, n, m)
    b2_mse = v5.cont_mse(pred_b2, X_eval)

    n_int_full = int(rich_full["intervention_mask"].sum())
    print(f"  Seed {seed}: n_int_full={n_int_full}, K_true_edges={len(true_edges)}")

    results = []
    for p in p_ints:
        rich_sub = subsample_rich(rich_full, p, seed=seed)
        n_int_sub = int(rich_sub["intervention_mask"].sum())
        # ICP F-test on subsampled
        if n_int_sub < 5:
            # too few interventions — return zero edges
            icp = {"validated_edges": [], "details": {}}
        else:
            icp = v5.icp_validate_all(rich_sub["X"], rich_sub["A"],
                                      rich_sub["intervention_mask"],
                                      K, n, m, alpha=0.05)
        val_set = set((int(i), int(j)) for (i, j) in icp["validated_edges"])
        true_set = set(true_edges)
        tp = len(val_set & true_set)
        fp = len(val_set - true_set)
        fn = len(true_set - val_set)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        # Fit B8d with these parents
        parents_b8 = {k: [] for k in range(K)}
        for (i, j) in icp["validated_edges"]:
            parents_b8[j].append(i)
        coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)
        pred_b8 = v2.predict_per_module(X_eval, A_eval, coefs_b8, parents_b8, lag=1)
        b8_mse = v5.cont_mse(pred_b8, X_eval)
        b8_em32 = v5.em_multi_bin(pred_b8, X_eval, X_train, n_bins_list=(32,))["em_32bin"]
        b2_em32 = v5.em_multi_bin(pred_b2, X_eval, X_train, n_bins_list=(32,))["em_32bin"]
        b10_em32 = v5.em_multi_bin(pred_b10, X_eval, X_train, n_bins_list=(32,))["em_32bin"]

        results.append({
            "p_int": p,
            "n_int_used": n_int_sub,
            "n_validated_edges": len(val_set),
            "n_true_edges": len(true_edges),
            "edge_f1": round(f1, 3),
            "edge_precision": round(prec, 3),
            "edge_recall": round(rec, 3),
            "B8d_MSE": round(b8_mse, 3),
            "B8d_EM32": round(b8_em32, 4),
            "B2_MSE": round(b2_mse, 3),
            "B10_MSE": round(b10_mse, 3),
            "B10_EM32": round(b10_em32, 4),
            "B2_EM32": round(b2_em32, 4),
        })
    return {"seed": seed, "n_int_full": n_int_full,
            "true_edges_count": len(true_edges),
            "K": K, "results": results}


# =========================================================
# G6: Spearman ρ of Δ_causal vs (d, α, γ)
# =========================================================

def gen_scm_with_params(cfg_base: dict, K=5, chain_depth=4, alpha=0.5,
                        gamma=0.5, gamma_A=2.0, lag_mean=0, sigma=0.10,
                        confound_action=True, seed=0,
                        T_train=5000, T_eval=2000) -> Dict[str, Any]:
    """Generate a fresh SCM with given parameter values + run B2/B10 + eval on confshift."""
    cfg = SCMConfig(
        K=K, n_k=4, m_k=2, rho=0.0, d=chain_depth,
        alpha=alpha, lag_mean=lag_mean, gamma=gamma, sigma=sigma,
        T_train=T_train, T_eval=T_eval, seed=seed,
        force_chain=True, chain_depth=chain_depth,
        confound_action=confound_action, gamma_A=gamma_A,
    )
    out = __import__("synthetic_scm_skeleton").generate_all_splits(cfg)
    oracle = out["oracle"]
    obs = out["splits"]["obs"]
    # confshift eval
    rng_cs = np.random.default_rng(seed * 31 + 23)
    ev_cs = simulate(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                     {"confounder_shift": True, "conf_mu": 2.5}, rng_cs, T=1500)
    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval = ev_cs["X"], ev_cs["A"]

    pred_b10 = v2.b10_predict(X_eval, A_eval, oracle, K, cfg.n_k)
    pred_b2 = v2.b2_fit_predict(X_train, A_train, X_eval, A_eval, K, cfg.n_k, cfg.m_k)
    b10_mse = v5.cont_mse(pred_b10, X_eval)
    b2_mse = v5.cont_mse(pred_b2, X_eval)
    delta_causal_mse = b2_mse - b10_mse  # positive = causal wins by this much
    # EM-based
    edges = v2.fit_bin_edges(X_train, n_bins=32)
    b2_em = v2.transition_em(v2.discretize(pred_b2, edges), v2.discretize(X_eval, edges))
    b10_em = v2.transition_em(v2.discretize(pred_b10, edges), v2.discretize(X_eval, edges))
    delta_causal_em = (b10_em - b2_em) * 100  # in pp
    return {"alpha": alpha, "gamma": gamma, "chain_depth": chain_depth,
            "seed": seed, "B2_MSE": b2_mse, "B10_MSE": b10_mse,
            "delta_causal_MSE": delta_causal_mse,
            "delta_causal_EM_pp": delta_causal_em}


def spearman_rho(xs, ys):
    """Compute Spearman ρ via numpy rank correlation."""
    xs = np.array(xs); ys = np.array(ys)
    n = len(xs)
    if n < 2: return 0.0
    rx = xs.argsort().argsort().astype(float)
    ry = ys.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def g6_sweep(seed: int = 0) -> Dict[str, Any]:
    """Sweep depth d, effect strength α, confounding γ; compute Spearman ρ."""
    print("\n=== G6 sweep: depth d ===")
    d_values = [1, 2, 3, 4, 6, 8]
    d_runs = []
    for d in d_values:
        # K must be >= d+1 for chain of depth d
        K = max(5, d + 1)
        r = gen_scm_with_params({}, K=K, chain_depth=min(d, K-1), alpha=0.5,
                                gamma=0.5, gamma_A=2.0, seed=seed,
                                T_train=3000, T_eval=1500)
        d_runs.append(r)
        print(f"  d={d}: Δ_causal_MSE = {r['delta_causal_MSE']:.3f}, Δ_causal_EM = {r['delta_causal_EM_pp']:.2f}pp")
    rho_d_mse = spearman_rho(d_values, [r['delta_causal_MSE'] for r in d_runs])
    rho_d_em = spearman_rho(d_values, [r['delta_causal_EM_pp'] for r in d_runs])
    print(f"  Spearman ρ(d → Δ_MSE) = {rho_d_mse:.3f}, ρ(d → Δ_EM) = {rho_d_em:.3f}")

    print("\n=== G6 sweep: effect strength α ===")
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    a_runs = []
    for a in alpha_values:
        r = gen_scm_with_params({}, K=5, chain_depth=4, alpha=a,
                                gamma=0.5, gamma_A=2.0, seed=seed,
                                T_train=3000, T_eval=1500)
        a_runs.append(r)
        print(f"  α={a}: Δ_causal_MSE = {r['delta_causal_MSE']:.3f}, Δ_causal_EM = {r['delta_causal_EM_pp']:.2f}pp")
    rho_a_mse = spearman_rho(alpha_values, [r['delta_causal_MSE'] for r in a_runs])
    rho_a_em = spearman_rho(alpha_values, [r['delta_causal_EM_pp'] for r in a_runs])
    print(f"  Spearman ρ(α → Δ_MSE) = {rho_a_mse:.3f}, ρ(α → Δ_EM) = {rho_a_em:.3f}")

    print("\n=== G6 sweep: confounding γ ===")
    gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    g_runs = []
    for g in gamma_values:
        # gamma_A scaled with gamma to keep backdoor magnitude
        r = gen_scm_with_params({}, K=5, chain_depth=4, alpha=0.5,
                                gamma=g, gamma_A=2.0 * max(g, 0.1), seed=seed,
                                T_train=3000, T_eval=1500)
        g_runs.append(r)
        print(f"  γ={g}: Δ_causal_MSE = {r['delta_causal_MSE']:.3f}, Δ_causal_EM = {r['delta_causal_EM_pp']:.2f}pp")
    rho_g_mse = spearman_rho(gamma_values, [r['delta_causal_MSE'] for r in g_runs])
    rho_g_em = spearman_rho(gamma_values, [r['delta_causal_EM_pp'] for r in g_runs])
    print(f"  Spearman ρ(γ → Δ_MSE) = {rho_g_mse:.3f}, ρ(γ → Δ_EM) = {rho_g_em:.3f}")

    return {
        "d_sweep": {"values": d_values, "runs": d_runs,
                    "rho_MSE": rho_d_mse, "rho_EM": rho_d_em},
        "alpha_sweep": {"values": alpha_values, "runs": a_runs,
                        "rho_MSE": rho_a_mse, "rho_EM": rho_a_em},
        "gamma_sweep": {"values": gamma_values, "runs": g_runs,
                        "rho_MSE": rho_g_mse, "rho_EM": rho_g_em},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--out_dir", default="runs/exp7_g6_g7")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()

    # ============== G7 ==============
    print("=" * 90)
    print("G7: Intervention Coverage Scaling")
    print("=" * 90)
    p_ints = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
    g7_per_seed = []
    for s in args.seeds:
        print(f"\nSeed {s}:")
        r = g7_run_seed(s, args.data_dir, p_ints)
        g7_per_seed.append(r)
        print(f"  {'p_int':>7} {'n_int':>7} {'n_edges':>9} {'edge_f1':>9} {'B8d_MSE':>10} {'B8d_EM32':>10} {'B2_MSE':>10}")
        for row in r["results"]:
            print(f"  {row['p_int']:>7.2f} {row['n_int_used']:>7} {row['n_validated_edges']:>9} {row['edge_f1']:>9.3f} {row['B8d_MSE']:>10.3f} {row['B8d_EM32']:>10.4f} {row['B2_MSE']:>10.3f}")

    # Aggregate G7
    print()
    print("=" * 90)
    print("G7 AGGREGATE (mean across seeds)")
    print("=" * 90)
    print(f"  {'p_int':>7} {'avg_n_int':>10} {'avg_edge_f1':>13} {'avg_B8d_MSE':>13} {'avg_B8d_EM32':>13}")
    g7_agg = []
    for i, p in enumerate(p_ints):
        n_ints = [s["results"][i]["n_int_used"] for s in g7_per_seed]
        f1s = [s["results"][i]["edge_f1"] for s in g7_per_seed]
        mses = [s["results"][i]["B8d_MSE"] for s in g7_per_seed]
        ems = [s["results"][i]["B8d_EM32"] for s in g7_per_seed]
        row = {
            "p_int": p,
            "avg_n_int": round(np.mean(n_ints), 1),
            "avg_edge_f1": round(np.mean(f1s), 3),
            "avg_B8d_MSE": round(np.mean(mses), 3),
            "avg_B8d_EM32": round(np.mean(ems), 4),
        }
        g7_agg.append(row)
        print(f"  {row['p_int']:>7.2f} {row['avg_n_int']:>10.1f} {row['avg_edge_f1']:>13.3f} {row['avg_B8d_MSE']:>13.3f} {row['avg_B8d_EM32']:>13.4f}")

    # Check predictions:
    # (a) p=0: edge_f1 ≈ 0 (no intervention evidence)
    # (b) p=0.05: edge_f1 jumps if N >= N_min
    # (c) p=0.25: ≥50% of full advantage
    # (d) p>=0.50: marginal ≤ 1pp (saturation)
    print()
    print("=== G7 Theorem 2 / Saturation analysis ===")
    f1_p0 = g7_agg[0]["avg_edge_f1"]
    f1_p005 = g7_agg[1]["avg_edge_f1"]
    f1_p025 = g7_agg[3]["avg_edge_f1"]
    f1_p10 = g7_agg[5]["avg_edge_f1"]
    print(f"  edge_f1 at p=0: {f1_p0} (predict ≈ 0)")
    print(f"  edge_f1 at p=0.05: {f1_p005} (predict jump if N_min crossed)")
    print(f"  edge_f1 at p=0.25: {f1_p025} ({100 * f1_p025 / max(f1_p10, 0.01):.0f}% of full)")
    print(f"  edge_f1 at p=1.0: {f1_p10} (full advantage)")
    sat_check = abs(g7_agg[4]["avg_edge_f1"] - g7_agg[5]["avg_edge_f1"])
    print(f"  Saturation check: |edge_f1(p=0.5) - edge_f1(p=1)| = {sat_check:.3f} (saturated if ≤ 0.05)")

    # ============== G6 ==============
    print()
    print("=" * 90)
    print("G6: Spearman ρ of Δ_causal vs (d, α, γ)")
    print("=" * 90)
    g6_result = g6_sweep(seed=0)

    print()
    print("=" * 90)
    print("G6 SUMMARY")
    print("=" * 90)
    print(f"  Spearman ρ(d → Δ_causal_MSE):  {g6_result['d_sweep']['rho_MSE']:+.3f}  (gate ≥ 0.7)")
    print(f"  Spearman ρ(α → Δ_causal_MSE):  {g6_result['alpha_sweep']['rho_MSE']:+.3f}  (gate ≥ 0.7)")
    print(f"  Spearman ρ(γ → Δ_causal_MSE):  {g6_result['gamma_sweep']['rho_MSE']:+.3f}  (gate ≥ 0.7)")
    print()
    print(f"  Spearman ρ(d → Δ_causal_EM):   {g6_result['d_sweep']['rho_EM']:+.3f}")
    print(f"  Spearman ρ(α → Δ_causal_EM):   {g6_result['alpha_sweep']['rho_EM']:+.3f}")
    print(f"  Spearman ρ(γ → Δ_causal_EM):   {g6_result['gamma_sweep']['rho_EM']:+.3f}")

    rho_pass_count = sum([
        g6_result['d_sweep']['rho_MSE'] >= 0.7,
        g6_result['alpha_sweep']['rho_MSE'] >= 0.7,
        g6_result['gamma_sweep']['rho_MSE'] >= 0.7,
    ])
    print()
    print(f"G6 result: {rho_pass_count}/3 dimensions PASS Spearman ρ ≥ 0.7 (need ≥1 per design spec d/α/γ critical)")

    elapsed = time.time() - t0
    summary = {
        "phase": "exp7_G6_G7",
        "elapsed_seconds": elapsed,
        "G7": {"p_ints": p_ints, "per_seed": g7_per_seed, "aggregate": g7_agg,
               "saturation_check": sat_check < 0.05},
        "G6": g6_result,
        "G6_rho_pass_count_MSE": rho_pass_count,
    }
    out_path = os.path.join(args.out_dir, "exp7_g6_g7_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
