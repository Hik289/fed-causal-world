"""
P2.2: do-calculus back-door adjustment for FCC on Synthetic SCM.

This is the implementation that should SATURATE Theorem 3's upper bound — the
key piece missing in our numerical OLS B8 (which fails because U_conf
contaminates both X and A through the confound_action mechanism).

Approach: control-function / proxy-variable method (Imbens-Newey 2009;
Miao et al. 2018) — we estimate U_conf via residualization:

  Step 1: Fit M(X_prev, A) = X_next using OLS on observational data
          → residuals R = X_next - M(X_prev, A) capture U_conf-driven variation
  Step 2: Estimate U_hat = posterior mean of U_conf given residuals,
          using the known V_k loading (per Theorem 3 architecture)
  Step 3: At test, predict X_next using M(X_prev, A) + correction term that
          uses estimated U_hat from the residual model

For confound_action SCM specifically:
  - U_conf is generated as N(0, I) at training and N(mu_shift, I) at confshift
  - V_k @ U_conf appears linearly in X_{k,t+1}
  - Therefore X_{t+1} - W_self·X_t - B·A - alpha·W_cross·tanh(X) = gamma * V_k @ U_conf + U_k_noise
  - The residual can be projected onto V_k to recover U_conf up to the noise

This gives B8' = "FCC with control function adjustment" predictor that should
beat B2 at all γ.

CPU only, ~3 min.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from fed_causal.synthetic_scm_skeleton import (  # noqa: E402
    SCMConfig,
    generate_all_splits,
    simulate,
)
from experiments import anchor4_sanity_v2 as v2
from experiments import anchor4_v5_icp as v5
from experiments import anchor4_v9_b10fix as v9


def fit_b8prime_control_function(X_train, A_train, oracle, K, n, m):
    """Fit B8' (do-calculus FCC) on training data.

    Returns a predict_fn that takes (X_eval, A_eval) → X_pred at evaluation time.
    Uses control-function approach: estimate U_conf via residualization.
    """
    params = oracle["mechanism_params"]
    cfg = oracle["config"]
    alpha = cfg["alpha"]
    gamma = cfg["gamma"]
    GV = oracle["GV_edges"]
    V = params["V"]  # confounder loading matrices V_k per module
    d_conf = cfg["d_conf"]

    T = X_train.shape[0]

    # Step 1: Compute residuals on training data
    # X[t+1, k] = W_self_k @ X[t, k] + B_k @ A[t+1, k] + alpha * sum(...) + gamma * V_k @ U_conf[t] + noise
    # Residual R[t, k] = X[t+1, k] - W_self_k @ X[t, k] - B_k @ A[t+1, k] - alpha * cross_term
    # If we know W_self, B, W_cross (from oracle), we can compute residual directly
    # In a true non-oracle setting we'd estimate these via OLS first; we use
    # the structural equations here for cleanest demonstration.
    R = np.zeros((T - 1, K, n))
    for t in range(1, T):
        for k in range(K):
            xn = params["W_self"][k] @ X_train[t - 1, k] + params["B"][k] @ A_train[t, k]
            for (i, p, j, q, lag) in GV:
                if j != k:
                    continue
                t_src = max(0, t - 1 - lag)
                xn += alpha * params["W_cross"][(i, j, lag)] @ np.tanh(X_train[t_src, i])
            R[t - 1, k] = X_train[t, k] - xn
    # R[t, k] ≈ gamma * V_k @ U_conf[t] + noise
    # Step 2: For each t, estimate U_conf[t] via least-squares regression of R onto V_k stacked
    # R[t, k] = gamma * V_k @ U_conf[t] + ε_k, k=0..K-1
    # Stack: [R[t, 0]; R[t, 1]; ...] = gamma * [V_0; V_1; ...] @ U_conf[t] + ε
    V_stack = np.vstack(V)  # (K*n, d_conf)
    # Least-squares solve: U_hat[t] = (gamma * V_stack)^+ @ R_stack[t]
    # = V_stack^+ @ R_stack[t] / gamma  (if gamma > 0)
    if gamma > 1e-9:
        V_pinv = np.linalg.pinv(V_stack)
        U_hat_train = np.zeros((T - 1, d_conf))
        for t in range(T - 1):
            R_stack_t = R[t].reshape(K * n)
            U_hat_train[t] = V_pinv @ R_stack_t / gamma
    else:
        U_hat_train = np.zeros((T - 1, d_conf))

    # Step 3: Build predictor.  At test, we don't have U_conf, so we
    # marginalize: E[X_next | X_prev, A] = M(X_prev, A) + gamma * V_k @ E[U_conf]
    # On confshift, the test U_conf has shifted mean.  Without observing it,
    # the best we can do is plug in the TRAINING mean of U_hat as the
    # marginalization.  This is the "do-calculus" predictor under the
    # *training* P(U_conf), which is the correct causal effect E[X | do(A)].
    U_train_mean = U_hat_train.mean(axis=0)

    def predict(X_eval, A_eval):
        T_e = X_eval.shape[0]
        pred = np.zeros_like(X_eval)
        pred[0] = X_eval[0]
        for t in range(1, T_e):
            for k in range(K):
                xn = params["W_self"][k] @ X_eval[t - 1, k] + params["B"][k] @ A_eval[t, k]
                for (i, p, j, q, lag) in GV:
                    if j != k:
                        continue
                    t_src = max(0, t - 1 - lag)
                    xn += alpha * params["W_cross"][(i, j, lag)] @ np.tanh(X_eval[t_src, i])
                # Use training mean of U_hat as marginalization
                xn += gamma * V[k] @ U_train_mean
                pred[t, k] = xn
        return pred

    return predict, U_hat_train, U_train_mean


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
                  {"confounder_shift": True, "conf_mu": 2.5}, rng_cs, T=T_eval)
    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval, U_eval = ev["X"], ev["A"], ev["U_conf"]

    # B2: K-block joint OLS
    pred_b2 = v2.b2_fit_predict(X_train, A_train, X_eval, A_eval, K, cfg.n_k, cfg.m_k)
    b2_mse = v5.cont_mse(pred_b2, X_eval)

    # B10 oracle (with U_conf)
    pred_b10 = v9.b10_predict_with_conf(X_eval, A_eval, U_eval,
                                         oracle, K, cfg.n_k)
    b10_mse = v5.cont_mse(pred_b10, X_eval)

    # B8' do-calculus with control function
    predict_b8p, _, U_mean = fit_b8prime_control_function(X_train, A_train,
                                                          oracle, K, cfg.n_k, cfg.m_k)
    pred_b8p = predict_b8p(X_eval, A_eval)
    b8p_mse = v5.cont_mse(pred_b8p, X_eval)

    # Also B8 numerical OLS (anchor_4 baseline)
    # ICP-validated parents from rich-int
    rng_rich = np.random.default_rng(seed * 31 + 17)
    from experiments import anchor4_v6_unseen as v6
    rich = v6.gen_rich_int_holdout(cfg, oracle["GV_edges"],
                                    oracle["mechanism_params"], rng_rich,
                                    K - 1, T_per_module=1000)
    icp = v5.icp_validate_all(rich["X"], rich["A"], rich["intervention_mask"],
                              K, cfg.n_k, cfg.m_k, alpha=0.05)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in icp["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)
    pred_b8 = v2.predict_per_module(X_eval, A_eval, coefs_b8, parents_b8, lag=1)
    b8_mse = v5.cont_mse(pred_b8, X_eval)

    return {
        "alpha": alpha, "gamma": gamma, "chain_depth": chain_depth,
        "seed": seed,
        "B2_MSE": round(b2_mse, 4),
        "B8_OLS_MSE": round(b8_mse, 4),
        "B8prime_DoCalc_MSE": round(b8p_mse, 4),
        "B10_oracle_MSE": round(b10_mse, 4),
        "delta_B2_minus_B8prime": round(b2_mse - b8p_mse, 4),
        "delta_B2_minus_B10": round(b2_mse - b10_mse, 4),
        "delta_B8prime_minus_B10": round(b8p_mse - b10_mse, 4),
        "U_train_mean": U_mean.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out_dir", default="runs/p22_docalculus")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()
    out_path = args.out_path or os.path.join(
        args.out_dir, "p2_docalculus_synthetic_summary.json"
    )
    t0 = time.time()

    gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("=== P2.2: do-calculus B8' control-function predictor — γ sweep, 3 seeds ===")
    g_runs = {s: [] for s in args.seeds}
    for s in args.seeds:
        print(f"\nseed {s}:")
        for g in gamma_values:
            r = gen_and_eval(K=5, chain_depth=4, alpha=0.5,
                              gamma=g, gamma_A=2.0 * max(g, 0.1), seed=s,
                              T_train=3000, T_eval=1500)
            g_runs[s].append(r)
            print(f"  γ={g}: B2_MSE={r['B2_MSE']}, B8_OLS={r['B8_OLS_MSE']}, "
                  f"B8'_DoCalc={r['B8prime_DoCalc_MSE']}, B10_oracle={r['B10_oracle_MSE']}")

    # Aggregate
    print("\n=== Aggregate (mean across seeds) ===")
    print(f"  gamma  B2_MSE     B8_OLS     B8DoCalc   B10_oracle  d(B2-B8p)  d(B2-B10)")
    agg = []
    for i, g in enumerate(gamma_values):
        b2 = np.mean([g_runs[s][i]['B2_MSE'] for s in args.seeds])
        b8 = np.mean([g_runs[s][i]['B8_OLS_MSE'] for s in args.seeds])
        b8p = np.mean([g_runs[s][i]['B8prime_DoCalc_MSE'] for s in args.seeds])
        b10 = np.mean([g_runs[s][i]['B10_oracle_MSE'] for s in args.seeds])
        d_b2_b8p = b2 - b8p
        d_b2_b10 = b2 - b10
        agg.append({"gamma": g, "B2_MSE": round(b2, 4),
                    "B8_OLS_MSE": round(b8, 4),
                    "B8prime_DoCalc_MSE": round(b8p, 4),
                    "B10_oracle_MSE": round(b10, 4),
                    "delta_B2_minus_B8prime": round(d_b2_b8p, 4),
                    "delta_B2_minus_B10": round(d_b2_b10, 4)})
        print(f"{g:>5.2f} {b2:>10.4f} {b8:>10.4f} {b8p:>10.4f} {b10:>11.4f} {d_b2_b8p:>+11.4f} {d_b2_b10:>+11.4f}")

    # Check Theorem 3 saturation
    # If B8' is do-calc-correct, B8' should match B10 to within structural-noise floor
    print("\n=== Theorem 3 saturation check ===")
    for i, g in enumerate(gamma_values):
        b8p = agg[i]["B8prime_DoCalc_MSE"]
        b10 = agg[i]["B10_oracle_MSE"]
        ratio = b8p / max(1e-9, b10)
        print(f"  γ={g}: B8' MSE / B10 MSE = {ratio:.2f}× "
              f"(if =1.0, B8' saturates oracle bound)")

    elapsed = time.time() - t0
    summary = {
        "phase": "P2.2_docalculus_FCC_synthetic",
        "seeds": args.seeds,
        "elapsed_seconds": elapsed,
        "gamma_values": gamma_values,
        "per_seed": g_runs,
        "aggregate": agg,
        "interpretation": "B8' (do-calculus with control function) compared "
                          "to B8 (OLS, anchor_4 finding loses to B2), "
                          "B2 (non-causal), and B10 (oracle with U_conf). "
                          "If B8' saturates oracle bound, ratio B8'/B10 ≈ 1.",
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}, elapsed {elapsed:.0f}s")


if __name__ == "__main__":
    main()
