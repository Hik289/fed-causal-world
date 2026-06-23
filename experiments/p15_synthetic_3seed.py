"""
P1.5: Multi-seed Synthetic G6 sweep (γ, α, d).

exp7_g6_v2 was single-seed (seed=0). This re-runs with 3 seeds {0,1,2}
to provide robustness for paper §3 Theorem 1 verification.

CPU only, ~10 min.
"""

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, "/home/user/fedcausalworld/data")
sys.path.insert(0, "/home/user/fedcausalworld/experiments")
from synthetic_scm_skeleton import SCMConfig, simulate, generate_all_splits
import anchor4_sanity_v2 as v2
import anchor4_v5_icp as v5
import anchor4_v9_b10fix as v9


def gen_and_eval(K, chain_depth, alpha, gamma, gamma_A, lag_mean=0,
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
    ev = simulate(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                  {"confounder_shift": True, "conf_mu": 2.5}, rng_cs, T=1500)
    pred_b10 = v9.b10_predict_with_conf(ev["X"], ev["A"], ev["U_conf"],
                                         oracle, K, cfg.n_k)
    pred_b2 = v2.b2_fit_predict(obs["X"], obs["A"], ev["X"], ev["A"],
                                K, cfg.n_k, cfg.m_k)
    b10_mse = v5.cont_mse(pred_b10, ev["X"])
    b2_mse = v5.cont_mse(pred_b2, ev["X"])
    return {
        "alpha": alpha, "gamma": gamma, "chain_depth": chain_depth,
        "seed": seed,
        "B2_MSE": round(b2_mse, 4), "B10_MSE": round(b10_mse, 4),
        "delta_causal_MSE": round(b2_mse - b10_mse, 4),
    }


def spearman_rho(xs, ys):
    xs = np.array(xs); ys = np.array(ys)
    n = len(xs)
    if n < 2: return 0.0
    rx = xs.argsort().argsort().astype(float)
    ry = ys.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out_path",
                    default="/home/user/fedcausalworld/experiments/anchor_4_run/p1_synthetic_3seed_summary.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=== P1.5: 3-seed Synthetic G6 (γ, α, d sweep) ===")

    d_values = [1, 2, 3, 4, 6, 8]
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    # γ sweep (most important)
    print("\n-- γ sweep --")
    g_runs_per_seed = {}
    for s in args.seeds:
        g_runs_per_seed[s] = []
        for g in gamma_values:
            r = gen_and_eval(K=5, chain_depth=4, alpha=0.5,
                              gamma=g, gamma_A=2.0 * max(g, 0.1),
                              seed=s, T_train=3000, T_eval=1500)
            g_runs_per_seed[s].append(r)
            print(f"  seed {s} γ={g}: Δ={r['delta_causal_MSE']:.4f}")

    # Aggregate per γ across seeds
    gamma_agg = []
    for i, g in enumerate(gamma_values):
        ds = [g_runs_per_seed[s][i]['delta_causal_MSE'] for s in args.seeds]
        gamma_agg.append({"gamma": g, "mean": float(np.mean(ds)),
                          "std": float(np.std(ds))})
    gamma_means = [a['mean'] for a in gamma_agg]
    rho_g = spearman_rho(gamma_values, gamma_means)
    print(f"\nγ: aggregate means {[round(m, 3) for m in gamma_means]}, ρ={rho_g:+.3f}")

    # α sweep
    print("\n-- α sweep --")
    a_runs_per_seed = {}
    for s in args.seeds:
        a_runs_per_seed[s] = []
        for a in alpha_values:
            r = gen_and_eval(K=5, chain_depth=4, alpha=a,
                              gamma=0.5, gamma_A=2.0,
                              seed=s, T_train=3000, T_eval=1500)
            a_runs_per_seed[s].append(r)
        print(f"  seed {s} done")
    alpha_agg = []
    for i, a in enumerate(alpha_values):
        ds = [a_runs_per_seed[s][i]['delta_causal_MSE'] for s in args.seeds]
        alpha_agg.append({"alpha": a, "mean": float(np.mean(ds)),
                          "std": float(np.std(ds))})
    alpha_means = [a['mean'] for a in alpha_agg]
    rho_a = spearman_rho(alpha_values, alpha_means)
    print(f"α: aggregate means {[round(m, 3) for m in alpha_means]}, ρ={rho_a:+.3f}")

    # d sweep
    print("\n-- d sweep --")
    d_runs_per_seed = {}
    for s in args.seeds:
        d_runs_per_seed[s] = []
        for d in d_values:
            K = max(5, d + 1)
            r = gen_and_eval(K=K, chain_depth=d, alpha=0.5,
                              gamma=0.5, gamma_A=2.0,
                              seed=s, T_train=3000, T_eval=1500)
            d_runs_per_seed[s].append(r)
        print(f"  seed {s} done")
    d_agg = []
    for i, d in enumerate(d_values):
        ds = [d_runs_per_seed[s][i]['delta_causal_MSE'] for s in args.seeds]
        d_agg.append({"chain_depth": d, "mean": float(np.mean(ds)),
                      "std": float(np.std(ds))})
    d_means = [a['mean'] for a in d_agg]
    rho_d = spearman_rho(d_values, d_means)
    print(f"d: aggregate means {[round(m, 3) for m in d_means]}, ρ={rho_d:+.3f}")

    elapsed = time.time() - t0
    summary = {
        "phase": "P1.5_synthetic_3seed_G6",
        "seeds": args.seeds,
        "elapsed_seconds": elapsed,
        "gamma_sweep": {"values": gamma_values, "per_seed": g_runs_per_seed,
                        "aggregate": gamma_agg, "spearman_rho": rho_g,
                        "gate_pass_0.7": rho_g >= 0.7},
        "alpha_sweep": {"values": alpha_values, "per_seed": a_runs_per_seed,
                        "aggregate": alpha_agg, "spearman_rho": rho_a,
                        "gate_pass_0.7": rho_a >= 0.7},
        "d_sweep": {"values": d_values, "per_seed": d_runs_per_seed,
                    "aggregate": d_agg, "spearman_rho": rho_d,
                    "gate_pass_0.7": rho_d >= 0.7},
        "G6_n_pass": int((rho_g >= 0.7) + (rho_a >= 0.7) + (rho_d >= 0.7)),
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {args.out_path}, elapsed {elapsed:.0f}s")
    print(f"G6 verdict: ρ_γ={rho_g:+.3f}, ρ_α={rho_a:+.3f}, ρ_d={rho_d:+.3f}")
    print(f"G6 n_pass (≥0.7): {summary['G6_n_pass']}/3")


if __name__ == "__main__":
    main()
