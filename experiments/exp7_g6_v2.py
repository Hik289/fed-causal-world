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
from experiments import anchor4_sanity_v2 as v2
from experiments import anchor4_v5_icp as v5
from experiments import anchor4_v6_unseen as v6
from experiments import anchor4_v9_b10fix as v9
from fed_causal.synthetic_scm_skeleton import (
    SCMConfig,
    generate_all_splits,
    simulate,
)


def gen_and_eval_v9_style(K, chain_depth, alpha, gamma, gamma_A, lag_mean=0,
                          sigma=0.10, T_train=3000, T_eval=1500, seed=0):
    cfg = SCMConfig(
        K=K, n_k=4, m_k=2, rho=0.0, d=chain_depth,
        alpha=alpha, lag_mean=lag_mean, gamma=gamma, sigma=sigma,
        T_train=T_train, T_eval=T_eval, seed=seed,
        force_chain=True, chain_depth=min(chain_depth, K - 1),
        confound_action=True, gamma_A=gamma_A,
    )
    out = generate_all_splits(cfg)
    oracle = out["oracle"]
    obs = out["splits"]["obs"]
    rng_cs = np.random.default_rng(seed * 31 + 23)
    ev_cs = simulate(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                     {"confounder_shift": True, "conf_mu": 2.5}, rng_cs, T=T_eval)
    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval, U_eval = ev_cs["X"], ev_cs["A"], ev_cs["U_conf"]

    pred_b10 = v9.b10_predict_with_conf(X_eval, A_eval, U_eval, oracle, K, cfg.n_k)
    pred_b2 = v2.b2_fit_predict(X_train, A_train, X_eval, A_eval, K, cfg.n_k, cfg.m_k)
    b10_mse = v5.cont_mse(pred_b10, X_eval)
    b2_mse = v5.cont_mse(pred_b2, X_eval)
    return {
        "K": K, "chain_depth": chain_depth, "alpha": alpha, "gamma": gamma,
        "B2_MSE": round(b2_mse, 3), "B10_MSE": round(b10_mse, 3),
        "delta_causal_MSE": round(b2_mse - b10_mse, 3),
    }


def spearman_rho(xs, ys):
    xs = np.array(xs); ys = np.array(ys)
    n = len(xs)
    if n < 2: return 0.0
    rx = xs.argsort().argsort().astype(float)
    ry = ys.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/exp7_g6_v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 90)
    print("G6 v2: B10 with proper U_conf")
    print("=" * 90)

    print("\n=== G6 sweep: depth d ===")
    d_values = [1, 2, 3, 4, 6, 8]
    d_runs = []
    for d in d_values:
        K = max(5, d + 1)
        r = gen_and_eval_v9_style(K=K, chain_depth=d, alpha=0.5,
                                  gamma=0.5, gamma_A=2.0, seed=args.seed)
        d_runs.append(r)
        print(f"  d={d}: B2_MSE={r['B2_MSE']}, B10_MSE={r['B10_MSE']}, Δ={r['delta_causal_MSE']:+.3f}")
    rho_d = spearman_rho(d_values, [r['delta_causal_MSE'] for r in d_runs])
    print(f"  Spearman ρ(d → Δ_causal_MSE) = {rho_d:+.3f}")

    print("\n=== G6 sweep: effect strength α ===")
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    a_runs = []
    for a in alpha_values:
        r = gen_and_eval_v9_style(K=5, chain_depth=4, alpha=a,
                                  gamma=0.5, gamma_A=2.0, seed=args.seed)
        a_runs.append(r)
        print(f"  α={a}: B2_MSE={r['B2_MSE']}, B10_MSE={r['B10_MSE']}, Δ={r['delta_causal_MSE']:+.3f}")
    rho_a = spearman_rho(alpha_values, [r['delta_causal_MSE'] for r in a_runs])
    print(f"  Spearman ρ(α → Δ_causal_MSE) = {rho_a:+.3f}")

    print("\n=== G6 sweep: confounding γ ===")
    gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    g_runs = []
    for g in gamma_values:
        r = gen_and_eval_v9_style(K=5, chain_depth=4, alpha=0.5,
                                  gamma=g, gamma_A=2.0 * max(g, 0.1),
                                  seed=args.seed)
        g_runs.append(r)
        print(f"  γ={g}: B2_MSE={r['B2_MSE']}, B10_MSE={r['B10_MSE']}, Δ={r['delta_causal_MSE']:+.3f}")
    rho_g = spearman_rho(gamma_values, [r['delta_causal_MSE'] for r in g_runs])
    print(f"  Spearman ρ(γ → Δ_causal_MSE) = {rho_g:+.3f}")

    print()
    print("=" * 90)
    print("G6 SUMMARY (B10 with U_conf, properly testing Theorem 1)")
    print("=" * 90)
    print(f"  Spearman ρ(d → Δ_causal):  {rho_d:+.3f}  (gate ≥ 0.7)")
    print(f"  Spearman ρ(α → Δ_causal):  {rho_a:+.3f}  (gate ≥ 0.7)")
    print(f"  Spearman ρ(γ → Δ_causal):  {rho_g:+.3f}  (gate ≥ 0.7)")
    n_pass = sum([rho_d >= 0.7, rho_a >= 0.7, rho_g >= 0.7])
    print(f"\n  G6 result: {n_pass}/3 dimensions PASS")

    out = {
        "phase": "exp7_G6_v2_fixed_oracle",
        "d_sweep": {"values": d_values, "runs": d_runs, "spearman_rho": rho_d},
        "alpha_sweep": {"values": alpha_values, "runs": a_runs, "spearman_rho": rho_a},
        "gamma_sweep": {"values": gamma_values, "runs": g_runs, "spearman_rho": rho_g},
        "G6_n_pass": n_pass,
    }
    out_path = os.path.join(args.out_dir, "exp7_g6_v2_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
