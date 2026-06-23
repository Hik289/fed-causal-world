"""
anchor4_v5_icp.py — H0.anchor_4 fix: Peters 2016 ICP-style edge validation +
continuous L2 MSE + 32-bin EM evaluation (P0 + P1 per user approval).

Replaces V2's binary `dX > threshold` counter with a regression-based test:

  For each candidate edge (i -> j), under do(A_i = a*) environment:
    Full model    : X[t+1, j] = β @ [X[t-1, all_K], A[t, all_K], 1] + noise
    Reduced model : same but drop the n_k columns for X[t-1, i]
    F-test        : F = ((RSS_red - RSS_full)/q) / (RSS_full/(n-p))
                    where q = n_k (columns removed), p = full feature count
  Decide:
    - Edge i->j VALIDATED iff F-test p-value < α_threshold (default 0.05)
    - This is the ICP invariance-violation test: under no edge i->j, dropping
      X[t-1, i] should not change RSS significantly when conditioned on
      do(A_i=*).  When edge exists, dropping i->j signal increases RSS.

Theorem 2 trackers (q_i, r_j, N_ij, N_min, P_verify) still computed in parallel
for hypothesis.md compatibility; gate criterion uses ICP-validated edges, not
P_verify (which can be inflated by spurious correlations).

Evaluation:
  - Continuous L2 MSE per module + global, ratios to B10
  - 5-bin EM (sanity comparison to V2)
  - 20-bin EM (P1 fine discretization)
  - 32-bin EM (P1 even finer)

Primary eval: V4 medium_chain γ=0.5 + confounder-shift split.
Sanity check: sanity_chain γ=0 + mediator-int split — expect B8 edge_f1 ≥ 0.95.
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
from scipy import stats as scipy_stats

sys.path.insert(0, "/home/user/fedcausalworld/data")
sys.path.insert(0, "/home/user/fedcausalworld/experiments")
from synthetic_scm_skeleton import SCMConfig, simulate                    # noqa: E402
import anchor4_sanity_v2 as v2                                             # noqa: E402


# ----------------------------------------------------------------------
# Data + eval generation
# ----------------------------------------------------------------------

def reload_cfg(d):
    return SCMConfig(**{k: v for k, v in d.items()
                        if k in SCMConfig.__dataclass_fields__})


def gen_rich_int(cfg, GV, params, rng, T_per_module=800):
    parts, masks = [], []
    for k in range(cfg.K):
        spec = {"do_module": k, "do_value": np.ones(cfg.m_k), "rate": 0.30}
        o = simulate(cfg, GV, params, spec, rng, T_per_module)
        parts.append(o)
        masks.append(o["intervention_mask"])
    return {
        "X": np.concatenate([p["X"] for p in parts], axis=0),
        "A": np.concatenate([p["A"] for p in parts], axis=0),
        "intervention_mask": np.concatenate(masks, axis=0),
        "per_module_offsets": np.cumsum([0] + [T_per_module] * cfg.K),
    }


def gen_confshift_eval(cfg, GV, params, rng, T=1500, mu=2.5):
    spec = {"confounder_shift": True, "conf_mu": mu}
    return simulate(cfg, GV, params, spec, rng, T)


def gen_mediator_eval(cfg, GV, params, rng, T=1500):
    spec = {"do_module": cfg.K // 2, "do_value": np.ones(cfg.m_k), "rate": 0.40}
    return simulate(cfg, GV, params, spec, rng, T)


# ----------------------------------------------------------------------
# ICP-style edge validation
# ----------------------------------------------------------------------

def fit_ols_rss(Phi: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> Tuple[float, int, int]:
    """Fit OLS via ridge solve; return (RSS, n_samples, n_features)."""
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    B = Phi.T @ Y
    W = np.linalg.solve(A, B)
    resid = Y - Phi @ W
    return float((resid ** 2).sum()), int(Phi.shape[0]), int(Phi.shape[1])


def icp_test_edge(X_train: np.ndarray, A_train: np.ndarray,
                  int_mask: np.ndarray, i: int, j: int, K: int, n: int, m: int,
                  lag: int = 1, ridge: float = 1e-3,
                  alpha: float = 0.05) -> Dict[str, Any]:
    """
    ICP-style F-test for edge (i -> j).

    Restrict to rows where module i was intervened (int_mask[t, i] == 1) plus
    a matched number of rows where NO intervention occurred.  This gives us
    the strongest signal for invariance break.

    Full model : X[t+1, j] = f(X[t-1, all K], A[t, all K])
    Reduced    : drop the n columns for module i
    F-test p < alpha  → edge VALIDATED
    """
    T = X_train.shape[0]
    # Pool rows: all t where int_mask[t, i] == 1, plus equal pool of no-int rows
    valid_t = np.arange(max(lag, 1), T - 1)
    idx_int = valid_t[int_mask[valid_t, i] == 1]
    idx_no = valid_t[int_mask[valid_t].sum(axis=1) == 0]
    if len(idx_int) < 30 or len(idx_no) < 30:
        return {"validated": False, "p_value": 1.0, "F": 0.0, "n_int": int(len(idx_int)),
                "reason": "insufficient_samples"}

    # Subsample no-int to size of int (balanced)
    n_pick = min(len(idx_int), len(idx_no))
    rng = np.random.default_rng(seed=(i * 1000 + j))
    idx_no_s = rng.choice(idx_no, size=n_pick, replace=False)
    idx = np.concatenate([idx_int, idx_no_s])

    # Build full design matrix [X[t-1, :, :] flat, A[t, :, :] flat, env_dummy, 1]
    Xprev_flat = X_train[idx - 1].reshape(len(idx), K * n)
    A_cur_flat = A_train[idx].reshape(len(idx), K * m)
    env = (int_mask[idx, i] == 1).astype(float)[:, None]
    Phi_full = np.concatenate([Xprev_flat, A_cur_flat, env, np.ones((len(idx), 1))],
                              axis=1)
    Y = X_train[idx + 1, j]                                                # (N, n)

    # Reduced model: drop module i's n columns from Xprev_flat
    col_start = i * n
    keep = [k for k in range(K * n) if k < col_start or k >= col_start + n]
    Xprev_red = Xprev_flat[:, keep]
    Phi_red = np.concatenate([Xprev_red, A_cur_flat, env, np.ones((len(idx), 1))],
                             axis=1)

    # Fit both, aggregate over n target dims (each dim → separate scalar RSS)
    rss_full_total, n_samples, p_full = 0.0, 0, 0
    rss_red_total, _, p_red = 0.0, 0, 0
    for q in range(n):
        rss_f, ns, pf = fit_ols_rss(Phi_full, Y[:, q], ridge)
        rss_r, _, pr = fit_ols_rss(Phi_red, Y[:, q], ridge)
        rss_full_total += rss_f
        rss_red_total += rss_r
        n_samples = ns
        p_full, p_red = pf, pr

    df_diff = (p_full - p_red) * n                                    # n target dims
    df_resid = (n_samples * n) - (p_full * n)
    if df_resid <= 0 or rss_full_total <= 0:
        return {"validated": False, "p_value": 1.0, "F": 0.0, "n_int": int(len(idx_int)),
                "reason": "degenerate_df"}
    F = ((rss_red_total - rss_full_total) / df_diff) / (rss_full_total / df_resid)
    if F < 0:
        F = 0.0
    p_value = 1.0 - scipy_stats.f.cdf(F, df_diff, df_resid)

    return {
        "validated": bool(p_value < alpha),
        "p_value": float(p_value),
        "F": float(F),
        "df_diff": int(df_diff), "df_resid": int(df_resid),
        "n_int": int(len(idx_int)),
        "rss_full": float(rss_full_total),
        "rss_red": float(rss_red_total),
    }


def icp_validate_all(X_train, A_train, int_mask, K, n, m, alpha=0.05,
                     ridge=1e-3) -> Dict[str, Any]:
    """Run ICP F-test for every candidate (i, j) with i != j."""
    edges = []
    p_values = np.ones((K, K))
    F_values = np.zeros((K, K))
    details = {}
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            t = icp_test_edge(X_train, A_train, int_mask, i, j, K, n, m,
                              alpha=alpha, ridge=ridge)
            p_values[i, j] = t["p_value"]
            F_values[i, j] = t["F"]
            details[f"{i}->{j}"] = t
            if t["validated"]:
                edges.append((int(i), int(j)))
    return {"validated_edges": edges,
            "p_values": p_values.tolist(),
            "F_values": F_values.tolist(),
            "details": details}


# ----------------------------------------------------------------------
# Theorem 2 trackers (in parallel; do NOT drive validation decision)
# ----------------------------------------------------------------------

def theorem2_trackers(int_mask, K) -> Dict[str, Any]:
    """Compute q_hat, N_for_p95 per hypothesis.md §2.2."""
    q_hat = np.array([int_mask[:, i].mean() for i in range(K)])
    q_pos = q_hat[q_hat > 0]
    q_min = float(q_pos.min()) if len(q_pos) else 0.0
    # r_min: we set this to 1.0 because ICP F-test directly verifies
    # interventional response; in the SCM continuous-state setup the
    # "observability" rate r_j is 1.0 by definition (every X[t] is observed).
    # We keep r_hat ≡ 1 for the analytic N_for_p95 estimate below.
    r_min = 1.0
    if q_min * r_min > 0:
        N_for_p95 = int(np.ceil(np.log(0.05) / np.log(1 - q_min * r_min)))
    else:
        N_for_p95 = float("inf")
    return {"q_hat": q_hat.tolist(), "q_min": q_min, "r_min": r_min,
            "N_for_p_verify_0_95": N_for_p95}


# ----------------------------------------------------------------------
# Continuous L2 MSE + multi-bin EM evaluation
# ----------------------------------------------------------------------

def discretize_with_edges(X, edges):
    return v2.discretize(X, edges)


def cont_mse(pred, true):
    """Mean squared L2 distance per timestep, averaged."""
    return float(((pred - true) ** 2).sum(axis=2).mean())


def cont_mse_per_module(pred, true):
    return {int(k): float(((pred[:, k] - true[:, k]) ** 2).sum(axis=1).mean())
            for k in range(pred.shape[1])}


def em_multi_bin(pred, true, X_train_for_edges, n_bins_list=(5, 20, 32)):
    out = {}
    for nb in n_bins_list:
        edges = v2.fit_bin_edges(X_train_for_edges, n_bins=nb)
        pd = v2.discretize(pred, edges)
        td = v2.discretize(true, edges)
        out[f"em_{nb}bin"] = float((pd == td).mean())
    return out


# ----------------------------------------------------------------------
# Run one seed: V5
# ----------------------------------------------------------------------

def run_seed_v5(seed: int, base_data_dir: str, config_prefix: str,
                eval_kind: str, alpha: float = 0.05) -> Dict[str, Any]:
    """eval_kind in {"confounder_shift", "mediator_int"}"""
    data_dir = os.path.join(base_data_dir, f"{config_prefix}_seed{seed}")
    obs = v2.load_npz(os.path.join(data_dir, "obs.npz"))
    with open(os.path.join(data_dir, "oracle.pkl"), "rb") as f:
        oracle = pickle.load(f)
    cfg = reload_cfg(oracle["config"])
    K, n, m = cfg.K, cfg.n_k, cfg.m_k

    rng = np.random.default_rng(seed * 31 + 17)
    rich = gen_rich_int(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                        rng, T_per_module=800)
    rng2 = np.random.default_rng(seed * 31 + 19)
    if eval_kind == "confounder_shift":
        ev = gen_confshift_eval(cfg, oracle["GV_edges"],
                                oracle["mechanism_params"], rng2, T=1500, mu=2.5)
    elif eval_kind == "mediator_int":
        ev = gen_mediator_eval(cfg, oracle["GV_edges"],
                               oracle["mechanism_params"], rng2, T=1500)
    else:
        raise ValueError(eval_kind)

    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval = ev["X"], ev["A"]

    # === Predictors ===
    pred_b10 = v2.b10_predict(X_eval, A_eval, oracle, K, n)
    pred_b2 = v2.b2_fit_predict(X_train, A_train, X_eval, A_eval, K, n, m)

    cand_obs = v2.b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = v2.fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)
    pred_b7 = v2.predict_per_module(X_eval, A_eval, coefs_b7, parents_b7, lag=1)

    # ICP-validated edges from rich-int training
    icp = icp_validate_all(rich["X"], rich["A"], rich["intervention_mask"],
                           K, n, m, alpha=alpha)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in icp["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)
    pred_b8 = v2.predict_per_module(X_eval, A_eval, coefs_b8, parents_b8, lag=1)

    # === Eval: continuous MSE + multi-bin EM ===
    metrics = {}
    for name, pred in [("B10", pred_b10), ("B2", pred_b2),
                       ("B7", pred_b7), ("B8", pred_b8)]:
        metrics[name] = {
            "mse": cont_mse(pred, X_eval),
            "mse_per_module": cont_mse_per_module(pred, X_eval),
            **em_multi_bin(pred, X_eval, X_train, n_bins_list=(5, 20, 32)),
        }

    # === Edge metrics ===
    GM = np.asarray(oracle["GM_adj"])
    true_edges = [(int(i), int(j)) for i in range(K) for j in range(K) if GM[i, j] > 0]
    val_set = set((int(i), int(j)) for (i, j) in icp["validated_edges"])
    tp = len(val_set & set(true_edges))
    fp = len(val_set - set(true_edges))
    fn = len(set(true_edges) - val_set)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    # B7's "edge set" (for reporting purposes)
    val_b7 = set((int(i), int(j)) for (i, j) in cand_obs)
    tp7 = len(val_b7 & set(true_edges))
    fp7 = len(val_b7 - set(true_edges))
    fn7 = len(set(true_edges) - val_b7)
    f1_b7 = (2 * (tp7 / max(1, tp7 + fp7)) * (tp7 / max(1, tp7 + fn7))
             / max(1e-9, tp7 / max(1, tp7 + fp7) + tp7 / max(1, tp7 + fn7)))

    th2 = theorem2_trackers(rich["intervention_mask"], K)

    return {
        "seed": seed, "K": K, "n_k": n, "m_k": m,
        "config_prefix": config_prefix, "eval_kind": eval_kind,
        "alpha_icp": alpha,
        "true_edges": true_edges,
        "icp_validated_edges": icp["validated_edges"],
        "b7_observational_edges": cand_obs,
        "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
        "edge_f1_b7": f1_b7,
        "metrics": metrics,
        "theorem2": th2,
        "icp_p_values_diag": icp["p_values"],
    }


def aggregate(per_seed, key_path):
    """key_path = ('metrics', 'B8', 'mse')"""
    vals = []
    for s in per_seed:
        d = s
        for k in key_path:
            d = d[k]
        vals.append(d)
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "values": vals}


# ----------------------------------------------------------------------
# Main: run BOTH sanity_chain and medium_chain
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/user/fedcausalworld/data/synthetic")
    ap.add_argument("--out_dir", default="/home/user/fedcausalworld/experiments/anchor_4_run")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    runs = []
    plan = [
        ("sanity_chain_d4", "mediator_int"),   # gamma=0 sanity
        ("medium_chain_d4", "confounder_shift"),  # gamma=0.5 primary V4 config
    ]
    for config_prefix, eval_kind in plan:
        print(f"\n=== {config_prefix} / eval={eval_kind} ===")
        per_seed = []
        for s in args.seeds:
            r = run_seed_v5(s, args.data_dir, config_prefix, eval_kind, alpha=args.alpha)
            per_seed.append(r)
            em32 = {n: r["metrics"][n]["em_32bin"] for n in ["B2", "B7", "B8", "B10"]}
            mse = {n: r["metrics"][n]["mse"] for n in ["B2", "B7", "B8", "B10"]}
            print(f"  seed {s}: EM_32bin "
                  f"B2={em32['B2']:.4f} B7={em32['B7']:.4f} "
                  f"B8={em32['B8']:.4f} B10={em32['B10']:.4f}")
            print(f"           MSE     "
                  f"B2={mse['B2']:.4f} B7={mse['B7']:.4f} "
                  f"B8={mse['B8']:.4f} B10={mse['B10']:.4f}")
            print(f"           edge_f1 B8={r['edge_f1']:.3f} (prec={r['edge_precision']:.2f} "
                  f"rec={r['edge_recall']:.2f}) B7={r['edge_f1_b7']:.3f}")
            print(f"           ICP validated: {r['icp_validated_edges']}")
            print(f"           True edges:    {r['true_edges']}")

        runs.append({"config_prefix": config_prefix, "eval_kind": eval_kind,
                     "per_seed": per_seed})

    elapsed = time.time() - t0

    # === Build aggregate table for both bin levels + MSE ===
    summary = {"phase": "RUNNING_anchor_4_v5_icp_regression",
               "alpha_icp": args.alpha,
               "elapsed_seconds": elapsed,
               "configs": []}

    for run in runs:
        per_seed = run["per_seed"]
        agg = {}
        for name in ["B2", "B7", "B8", "B10"]:
            agg[name] = {
                "mse": aggregate(per_seed, ("metrics", name, "mse")),
                "em_5bin": aggregate(per_seed, ("metrics", name, "em_5bin")),
                "em_20bin": aggregate(per_seed, ("metrics", name, "em_20bin")),
                "em_32bin": aggregate(per_seed, ("metrics", name, "em_32bin")),
            }
        edge_f1 = aggregate(per_seed, ("edge_f1",))
        # Gate evaluation against multiple metrics (user invariant unchanged)
        B8_mse = agg["B8"]["mse"]["mean"]
        B10_mse = agg["B10"]["mse"]["mean"]
        B2_mse = agg["B2"]["mse"]["mean"]
        # Use 32-bin EM as the primary EM metric
        B8_em = agg["B8"]["em_32bin"]["mean"]
        B10_em = agg["B10"]["em_32bin"]["mean"]
        B2_em = agg["B2"]["em_32bin"]["mean"]

        gate = {
            "primary_metric": "transition_em_32bin",
            "B10_em32": round(B10_em * 100, 2),
            "B8_em32": round(B8_em * 100, 2),
            "B7_em32": round(agg["B7"]["em_32bin"]["mean"] * 100, 2),
            "B2_em32": round(B2_em * 100, 2),
            "B8_minus_B10_pp": round((B8_em - B10_em) * 100, 2),
            "B8_minus_B2_pp": round((B8_em - B2_em) * 100, 2),
            "criterion_1_B8_vs_B10_pass": (B8_em - B10_em) >= -0.03,
            "criterion_2_B8_vs_B2_pass": (B8_em - B2_em) >= 0.10,
            "secondary_metric": "continuous_l2_mse_ratio",
            "B8_mse": round(B8_mse, 4),
            "B10_mse": round(B10_mse, 4),
            "B2_mse": round(B2_mse, 4),
            "mse_ratio_B8_to_B2": round(B8_mse / max(1e-9, B2_mse), 4),
            "mse_ratio_B8_to_B10": round(B8_mse / max(1e-9, B10_mse), 4),
            "edge_f1_B8": round(edge_f1["mean"], 3),
        }
        gate["overall_pass"] = (gate["criterion_1_B8_vs_B10_pass"]
                                and gate["criterion_2_B8_vs_B2_pass"])
        summary["configs"].append({
            "config_prefix": run["config_prefix"],
            "eval_kind": run["eval_kind"],
            "per_seed": per_seed,
            "aggregate": agg,
            "edge_f1_b8_mean": edge_f1["mean"],
            "gate": gate,
        })

    # Primary gate is V4 (medium γ=0.5 + confounder-shift) per user
    primary = summary["configs"][1]
    sanity = summary["configs"][0]
    summary["primary_gate"] = primary["gate"]
    summary["primary_gate"]["overall_pass_primary_v4"] = primary["gate"]["overall_pass"]
    summary["sanity_edge_f1_B8"] = sanity["edge_f1_b8_mean"]
    summary["sanity_edge_f1_B7"] = float(np.mean([s["edge_f1_b7"] for s in sanity["per_seed"]]))
    summary["sanity_supplementary_pass"] = (sanity["edge_f1_b8_mean"] >= 0.95
                                            and summary["sanity_edge_f1_B7"] <= 0.5)

    out = os.path.join(args.out_dir, "anchor4_results_v5_icp.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}")
    print("=== PRIMARY GATE (V4 medium γ=0.5 + confounder-shift) ===")
    print(json.dumps(summary["primary_gate"], indent=2))
    print(f"\nSanity edge_f1 B8={summary['sanity_edge_f1_B8']:.3f} "
          f"(req ≥0.95), B7={summary['sanity_edge_f1_B7']:.3f} (req ≤0.5)")
    print(f"Sanity supplementary pass: {summary['sanity_supplementary_pass']}")

    return 0 if summary["primary_gate"]["overall_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
