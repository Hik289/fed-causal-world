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
from experiments import anchor4_sanity_v2 as v2
from experiments import anchor4_v5_icp as v5
from experiments import anchor4_v6_unseen as v6
from fed_causal.synthetic_scm_skeleton import SCMConfig, simulate


def gen_iid_eval(cfg, GV, params, rng, T=1500):
    return simulate(cfg, GV, params, {}, rng, T)


def gen_confshift_eval(cfg, GV, params, rng, T=1500, mu=2.5):
    spec = {"confounder_shift": True, "conf_mu": mu}
    return simulate(cfg, GV, params, spec, rng, T)


def gen_unseen_int_eval(cfg, GV, params, rng, k_holdout, T=1500):
    spec = {"do_module": k_holdout, "do_value": np.ones(cfg.m_k), "rate": 0.40}
    return simulate(cfg, GV, params, spec, rng, T)


def run_seed_v8(seed: int, base_data_dir: str, config_prefix: str = "confact_chain_d4",
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

    rng_iid = np.random.default_rng(seed * 31 + 21)
    ev_iid = gen_iid_eval(cfg, oracle["GV_edges"],
                          oracle["mechanism_params"], rng_iid, T=1500)
    rng_cs = np.random.default_rng(seed * 31 + 23)
    ev_cs = gen_confshift_eval(cfg, oracle["GV_edges"],
                               oracle["mechanism_params"], rng_cs,
                               T=1500, mu=2.5)
    rng_ui = np.random.default_rng(seed * 31 + 27)
    ev_ui = gen_unseen_int_eval(cfg, oracle["GV_edges"],
                                oracle["mechanism_params"], rng_ui,
                                k_holdout, T=1500)

    X_train, A_train = obs["X"], obs["A"]

    def b10_pred(X, A):
        return v2.b10_predict(X, A, oracle, K, n)

    def b2_pred(X, A):
        return v2.b2_fit_predict(X_train, A_train, X, A, K, n, m)

    cand_obs = v2.b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = v2.fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)

    def b7_pred(X, A):
        return v2.predict_per_module(X, A, coefs_b7, parents_b7, lag=1)

    icp = v5.icp_validate_all(rich["X"], rich["A"], rich["intervention_mask"],
                              K, n, m, alpha=alpha_icp)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in icp["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)

    def b8_pred(X, A):
        return v2.predict_per_module(X, A, coefs_b8, parents_b8, lag=1)

    splits = {"iid": (ev_iid["X"], ev_iid["A"]),
              "confshift": (ev_cs["X"], ev_cs["A"]),
              "unseen_int": (ev_ui["X"], ev_ui["A"])}

    results_per_split = {}
    for split_name, (X_e, A_e) in splits.items():
        preds = {"B10": b10_pred(X_e, A_e),
                 "B2":  b2_pred(X_e, A_e),
                 "B7":  b7_pred(X_e, A_e),
                 "B8":  b8_pred(X_e, A_e)}
        split_metrics = {}
        for name, pred in preds.items():
            split_metrics[name] = {
                "mse": v5.cont_mse(pred, X_e),
                "mse_per_module": v5.cont_mse_per_module(pred, X_e),
                **v5.em_multi_bin(pred, X_e, X_train, n_bins_list=(5, 20, 32)),
            }
        results_per_split[split_name] = split_metrics

    b2_iid_em32 = results_per_split["iid"]["B2"]["em_32bin"]
    b2_cs_em32 = results_per_split["confshift"]["B2"]["em_32bin"]
    b2_ui_em32 = results_per_split["unseen_int"]["B2"]["em_32bin"]
    delta_int_observed = {
        "B2_EM_drop_iid_to_confshift_pp": (b2_iid_em32 - b2_cs_em32) * 100,
        "B2_EM_drop_iid_to_unseen_int_pp": (b2_iid_em32 - b2_ui_em32) * 100,
    }

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
        "per_split": results_per_split,
        "delta_int_observed": delta_int_observed,
        "theorem2": v5.theorem2_trackers(rich["intervention_mask"], K),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--out_dir", default="runs/anchor_4_v8")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    per_seed = []
    for s in args.seeds:
        print(f"\n=== seed {s} (confact_chain_d4 with gamma_A=2.0) ===")
        r = run_seed_v8(s, args.data_dir)
        per_seed.append(r)
        print(f"  edge_f1={r['edge_f1']:.3f} validated={r['icp_validated_edges']}")
        print(f"  delta_int observed (B2 EM drop): "
              f"IID→conf_shift = {r['delta_int_observed']['B2_EM_drop_iid_to_confshift_pp']:.2f}pp  "
              f"IID→unseen_int = {r['delta_int_observed']['B2_EM_drop_iid_to_unseen_int_pp']:.2f}pp")
        for split_name in ["iid", "confshift", "unseen_int"]:
            spm = r["per_split"][split_name]
            em32 = {n: spm[n]["em_32bin"] for n in ["B2","B7","B8","B10"]}
            mse = {n: spm[n]["mse"] for n in ["B2","B7","B8","B10"]}
            print(f"  split={split_name:<11}"
                  f" EM32: B2={em32['B2']:.4f} B7={em32['B7']:.4f} "
                  f"B8={em32['B8']:.4f} B10={em32['B10']:.4f} "
                  f"(B8-B2={(em32['B8']-em32['B2'])*100:+.2f}pp)  "
                  f"MSE: B2={mse['B2']:.3f} B8={mse['B8']:.3f} B10={mse['B10']:.3f}")

    elapsed = time.time() - t0

    def agg(method, split, key):
        vals = [s["per_split"][split][method][key] for s in per_seed]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "phase": "RUNNING_anchor_4_v8_confact",
        "config": "confact_chain_d4 (K=5, chain=4, alpha=0.5, gamma=0.5, sigma=0.10, gamma_A=2.0, confound_action=True)",
        "elapsed_seconds": elapsed,
        "per_seed": per_seed,
        "aggregate_per_split": {},
        "delta_int_observed_mean": {
            "B2_EM_drop_iid_to_confshift_pp": float(np.mean(
                [s["delta_int_observed"]["B2_EM_drop_iid_to_confshift_pp"] for s in per_seed])),
            "B2_EM_drop_iid_to_unseen_int_pp": float(np.mean(
                [s["delta_int_observed"]["B2_EM_drop_iid_to_unseen_int_pp"] for s in per_seed])),
        },
        "edge_f1_mean": float(np.mean([s["edge_f1"] for s in per_seed])),
    }
    for split in ["iid", "confshift", "unseen_int"]:
        summary["aggregate_per_split"][split] = {}
        for m in ["B2", "B7", "B8", "B10"]:
            for k in ["em_5bin", "em_20bin", "em_32bin", "mse"]:
                summary["aggregate_per_split"][split].setdefault(m, {})
                summary["aggregate_per_split"][split][m][k] = agg(m, split, k)

    primary_split = "confshift"
    B10_em = summary["aggregate_per_split"][primary_split]["B10"]["em_32bin"]["mean"]
    B8_em = summary["aggregate_per_split"][primary_split]["B8"]["em_32bin"]["mean"]
    B7_em = summary["aggregate_per_split"][primary_split]["B7"]["em_32bin"]["mean"]
    B2_em = summary["aggregate_per_split"][primary_split]["B2"]["em_32bin"]["mean"]
    gate_primary = {
        "primary_split": primary_split,
        "primary_metric": "transition_em_32bin",
        "B10_em32_pp": round(B10_em * 100, 2),
        "B8_em32_pp": round(B8_em * 100, 2),
        "B7_em32_pp": round(B7_em * 100, 2),
        "B2_em32_pp": round(B2_em * 100, 2),
        "B8_minus_B10_pp": round((B8_em - B10_em) * 100, 2),
        "B8_minus_B2_pp": round((B8_em - B2_em) * 100, 2),
        "criterion_1_pass": (B8_em - B10_em) >= -0.03,
        "criterion_2_pass": (B8_em - B2_em) >= 0.10,
        "edge_f1_B8": summary["edge_f1_mean"],
        "delta_int_B2_drop_iid_to_confshift_pp":
            summary["delta_int_observed_mean"]["B2_EM_drop_iid_to_confshift_pp"],
        "delta_int_B2_drop_iid_to_unseen_int_pp":
            summary["delta_int_observed_mean"]["B2_EM_drop_iid_to_unseen_int_pp"],
    }
    gate_primary["overall_pass"] = (gate_primary["criterion_1_pass"]
                                    and gate_primary["criterion_2_pass"])

    for alt_split in ["unseen_int"]:
        B10a = summary["aggregate_per_split"][alt_split]["B10"]["em_32bin"]["mean"]
        B8a = summary["aggregate_per_split"][alt_split]["B8"]["em_32bin"]["mean"]
        B2a = summary["aggregate_per_split"][alt_split]["B2"]["em_32bin"]["mean"]
        summary[f"gate_alt_{alt_split}"] = {
            "B10_em32_pp": round(B10a*100, 2),
            "B8_em32_pp": round(B8a*100, 2),
            "B2_em32_pp": round(B2a*100, 2),
            "B8_minus_B10_pp": round((B8a-B10a)*100, 2),
            "B8_minus_B2_pp": round((B8a-B2a)*100, 2),
            "criterion_1_pass": (B8a - B10a) >= -0.03,
            "criterion_2_pass": (B8a - B2a) >= 0.10,
            "overall_pass": (B8a-B10a)>=-0.03 and (B8a-B2a)>=0.10,
        }

    summary["gate_primary"] = gate_primary

    out = os.path.join(args.out_dir, "anchor4_results_v8_confact.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}")
    print("=== GATE PRIMARY (confshift) ===")
    print(json.dumps(gate_primary, indent=2))
    print("=== GATE ALT (unseen_int) ===")
    print(json.dumps(summary["gate_alt_unseen_int"], indent=2))

    return 0 if gate_primary["overall_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
