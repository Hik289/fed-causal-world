"""
anchor4_v9_b10fix.py — V8 with corrected B10 (oracle now includes U_conf term).

V8 bug: B10 was missing the `gamma * V_k @ U_conf[t]` confounder contribution,
making B10 MSE > B2 MSE under confshift.  This patch:
  - B10 takes the eval split's stored U_conf array (already saved by simulate())
  - Adds gamma * V_k @ U_conf[t] to the rollout

Also re-fits eval-split bin edges from a combined (train + eval) pool so EM
discretization is not artificially saturated under confshift.

Two-branch reporting:
  - Pass:    B8 ≥ B10-3pp on confshift AND B8 ≥ B2+10pp on confshift|unseen_int
  - Fail-1:  B2 not degraded (delta_int_obs near 0) → genuine SCM design issue
  - Fail-2:  B2 degrades but B8 also degrades comparably → linear-Gaussian SCM
             is identifiable from obs by K-block OLS (Pearl 2009 §1.4) and
             cannot show causal-advantage in this regime
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import numpy as np

sys.path.insert(0, "/home/user/fedcausalworld/data")
sys.path.insert(0, "/home/user/fedcausalworld/experiments")
from synthetic_scm_skeleton import simulate                          # noqa: E402
import anchor4_sanity_v2 as v2                                        # noqa: E402
import anchor4_v5_icp as v5                                           # noqa: E402
import anchor4_v6_unseen as v6                                        # noqa: E402


def b10_predict_with_conf(X, A, U_conf, oracle, K, n) -> np.ndarray:
    """B10 oracle with U_conf included (full Assumption-2 oracle)."""
    params = oracle["mechanism_params"]
    cfg = oracle["config"]
    alpha = cfg["alpha"]
    gamma = cfg["gamma"]
    GV = oracle["GV_edges"]
    T = X.shape[0]
    pred = np.zeros_like(X)
    pred[0] = X[0]
    for t in range(1, T):
        for k in range(K):
            xn = params["W_self"][k] @ X[t - 1, k] + params["B"][k] @ A[t, k]
            for (i, p, j, q, lag) in GV:
                if j != k:
                    continue
                t_src = max(0, t - 1 - lag)
                xn += alpha * params["W_cross"][(i, j, lag)] @ np.tanh(X[t_src, i])
            # Confounder term (Assumption 2)
            xn += gamma * params["V"][k] @ U_conf[t]
            pred[t, k] = xn
    return pred


def fit_bins_from_pool(X_train, X_eval, n_bins=32):
    """Fit bins from a combined pool to avoid eval-shift saturation."""
    pool = np.concatenate([X_train, X_eval], axis=0)
    return v2.fit_bin_edges(pool, n_bins=n_bins)


def em_multi_bin_pooled(pred, true, X_train, X_eval, n_bins_list=(5, 20, 32)):
    out = {}
    for nb in n_bins_list:
        edges = fit_bins_from_pool(X_train, X_eval, n_bins=nb)
        pd = v2.discretize(pred, edges)
        td = v2.discretize(true, edges)
        out[f"em_{nb}bin_pooled"] = float((pd == td).mean())
    return out


def gen_iid_eval(cfg, GV, params, rng, T=1500):
    return simulate(cfg, GV, params, {}, rng, T)


def gen_confshift_eval(cfg, GV, params, rng, T=1500, mu=2.5):
    return simulate(cfg, GV, params, {"confounder_shift": True, "conf_mu": mu},
                    rng, T)


def gen_unseen_int_eval(cfg, GV, params, rng, k_holdout, T=1500):
    return simulate(cfg, GV, params,
                    {"do_module": k_holdout, "do_value": np.ones(cfg.m_k), "rate": 0.40},
                    rng, T)


def run_seed_v9(seed: int, base_data_dir: str,
                config_prefix: str = "confact_chain_d4",
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

    eval_splits = {}
    eval_splits["iid"] = gen_iid_eval(
        cfg, oracle["GV_edges"], oracle["mechanism_params"],
        np.random.default_rng(seed * 31 + 21), T=1500)
    eval_splits["confshift"] = gen_confshift_eval(
        cfg, oracle["GV_edges"], oracle["mechanism_params"],
        np.random.default_rng(seed * 31 + 23), T=1500, mu=2.5)
    eval_splits["unseen_int"] = gen_unseen_int_eval(
        cfg, oracle["GV_edges"], oracle["mechanism_params"],
        np.random.default_rng(seed * 31 + 27), k_holdout, T=1500)

    X_train, A_train = obs["X"], obs["A"]

    # === Predictors ===
    # B2 joint OLS (fit once on obs train)
    def b2_pred(X_e, A_e):
        return v2.b2_fit_predict(X_train, A_train, X_e, A_e, K, n, m)

    # B7 obs candidate edges
    cand_obs = v2.b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = v2.fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)

    def b7_pred(X_e, A_e):
        return v2.predict_per_module(X_e, A_e, coefs_b7, parents_b7, lag=1)

    # B8 ICP-validated
    icp = v5.icp_validate_all(rich["X"], rich["A"], rich["intervention_mask"],
                              K, n, m, alpha=alpha_icp)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in icp["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)

    def b8_pred(X_e, A_e):
        return v2.predict_per_module(X_e, A_e, coefs_b8, parents_b8, lag=1)

    # === Evaluate on each split (B10 uses U_conf from each split) ===
    per_split = {}
    for split_name, ev in eval_splits.items():
        X_e, A_e, U_e = ev["X"], ev["A"], ev["U_conf"]
        preds = {
            "B10": b10_predict_with_conf(X_e, A_e, U_e, oracle, K, n),
            "B2": b2_pred(X_e, A_e),
            "B7": b7_pred(X_e, A_e),
            "B8": b8_pred(X_e, A_e),
        }
        per_split[split_name] = {}
        for name, pred in preds.items():
            per_split[split_name][name] = {
                "mse": v5.cont_mse(pred, X_e),
                "mse_per_module": v5.cont_mse_per_module(pred, X_e),
                **v5.em_multi_bin(pred, X_e, X_train, n_bins_list=(5, 20, 32)),
                **em_multi_bin_pooled(pred, X_e, X_train, X_e,
                                       n_bins_list=(20, 32)),
            }

    # delta_int observed = B2 EM drop and MSE rise across splits
    delta = {}
    for tgt in ["confshift", "unseen_int"]:
        delta[f"B2_em32_pooled_drop_iid_to_{tgt}_pp"] = (
            per_split["iid"]["B2"]["em_32bin_pooled"]
            - per_split[tgt]["B2"]["em_32bin_pooled"]) * 100
        delta[f"B2_mse_rise_iid_to_{tgt}_ratio"] = (
            per_split[tgt]["B2"]["mse"] / per_split["iid"]["B2"]["mse"])

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
        "per_split": per_split,
        "delta_int_observed": delta,
        "theorem2": v5.theorem2_trackers(rich["intervention_mask"], K),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/user/fedcausalworld/data/synthetic")
    ap.add_argument("--out_dir", default="/home/user/fedcausalworld/experiments/anchor_4_run")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    per_seed = []
    for s in args.seeds:
        print(f"\n=== seed {s} V9 (confact_chain γ=0.5, γ_A=2.0, B10 with U_conf) ===")
        r = run_seed_v9(s, args.data_dir)
        per_seed.append(r)
        print(f"  edge_f1={r['edge_f1']:.3f} validated={r['icp_validated_edges']}")
        print(f"  delta_int_observed: {r['delta_int_observed']}")
        for split_name in ["iid", "confshift", "unseen_int"]:
            spm = r["per_split"][split_name]
            em32p = {n: spm[n]["em_32bin_pooled"] for n in ["B2","B7","B8","B10"]}
            mse = {n: spm[n]["mse"] for n in ["B2","B7","B8","B10"]}
            print(f"  {split_name:<11}"
                  f" EM32p: B2={em32p['B2']:.4f} B7={em32p['B7']:.4f} "
                  f"B8={em32p['B8']:.4f} B10={em32p['B10']:.4f} "
                  f"(B8-B2={(em32p['B8']-em32p['B2'])*100:+.2f}pp B8-B10={(em32p['B8']-em32p['B10'])*100:+.2f}pp)  "
                  f"MSE: B2={mse['B2']:.3f} B7={mse['B7']:.3f} B8={mse['B8']:.3f} B10={mse['B10']:.3f}")

    elapsed = time.time() - t0

    def agg(method, split, key):
        vals = [s["per_split"][split][method][key] for s in per_seed]
        return float(np.mean(vals))

    summary = {
        "phase": "RUNNING_anchor_4_v9_confact_b10fix",
        "config": "confact_chain_d4 (K=5, alpha=0.5, gamma=0.5, sigma=0.10, gamma_A=2.0, B10 with U_conf)",
        "elapsed_seconds": elapsed,
        "edge_f1_mean": float(np.mean([s["edge_f1"] for s in per_seed])),
        "per_seed": per_seed,
    }
    summary["aggregate_per_split"] = {}
    for split in ["iid", "confshift", "unseen_int"]:
        summary["aggregate_per_split"][split] = {}
        for method in ["B2", "B7", "B8", "B10"]:
            d = {}
            for key in ["mse", "em_5bin", "em_20bin", "em_32bin",
                        "em_20bin_pooled", "em_32bin_pooled"]:
                d[key] = agg(method, split, key)
            summary["aggregate_per_split"][split][method] = d
    summary["delta_int_observed_mean"] = {
        k: float(np.mean([s["delta_int_observed"][k] for s in per_seed]))
        for k in per_seed[0]["delta_int_observed"]
    }

    # GATE: primary=confshift on em_32bin_pooled
    gates = {}
    for split in ["confshift", "unseen_int"]:
        s = summary["aggregate_per_split"][split]
        B10 = s["B10"]["em_32bin_pooled"]
        B8 = s["B8"]["em_32bin_pooled"]
        B2 = s["B2"]["em_32bin_pooled"]
        gates[split] = {
            "B10_em32p_pp": round(B10 * 100, 2),
            "B8_em32p_pp": round(B8 * 100, 2),
            "B7_em32p_pp": round(s["B7"]["em_32bin_pooled"] * 100, 2),
            "B2_em32p_pp": round(B2 * 100, 2),
            "B8_minus_B10_pp": round((B8 - B10) * 100, 2),
            "B8_minus_B2_pp": round((B8 - B2) * 100, 2),
            "criterion_1_pass": (B8 - B10) >= -0.03,
            "criterion_2_pass": (B8 - B2) >= 0.10,
            "B10_MSE": s["B10"]["mse"],
            "B8_MSE": s["B8"]["mse"],
            "B2_MSE": s["B2"]["mse"],
            "MSE_ratio_B2_to_B10": s["B2"]["mse"] / max(1e-9, s["B10"]["mse"]),
            "MSE_ratio_B8_to_B10": s["B8"]["mse"] / max(1e-9, s["B10"]["mse"]),
        }
        gates[split]["overall_pass"] = (gates[split]["criterion_1_pass"]
                                        and gates[split]["criterion_2_pass"])
    summary["gates"] = gates

    out = os.path.join(args.out_dir, "anchor4_results_v9_b10fix.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}")
    print("=== GATE confshift ===")
    print(json.dumps(gates["confshift"], indent=2))
    print("=== GATE unseen_int ===")
    print(json.dumps(gates["unseen_int"], indent=2))
    return 0 if (gates["confshift"]["overall_pass"] or
                 gates["unseen_int"]["overall_pass"]) else 2


if __name__ == "__main__":
    sys.exit(main())
