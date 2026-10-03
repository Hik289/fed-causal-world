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
from fed_causal.synthetic_scm_skeleton import SCMConfig, simulate
from experiments import anchor4_sanity_v2 as v2
from experiments import anchor4_v5_icp as v5


def gen_rich_int_holdout(cfg, GV, params, rng, k_holdout: int,
                         T_per_module: int = 1000):
    parts, masks = [], []
    for k in range(cfg.K):
        if k == k_holdout:
            continue
        spec = {"do_module": k, "do_value": np.ones(cfg.m_k), "rate": 0.30}
        o = simulate(cfg, GV, params, spec, rng, T_per_module)
        parts.append(o)
        masks.append(o["intervention_mask"])
    return {
        "X": np.concatenate([p["X"] for p in parts], axis=0),
        "A": np.concatenate([p["A"] for p in parts], axis=0),
        "intervention_mask": np.concatenate(masks, axis=0),
    }


def gen_unseen_int_eval(cfg, GV, params, rng, k_holdout: int, T: int = 1500):
    spec = {"do_module": k_holdout, "do_value": np.ones(cfg.m_k), "rate": 0.40}
    return simulate(cfg, GV, params, spec, rng, T)


def run_seed_v6(seed: int, base_data_dir: str, config_prefix: str,
                alpha: float = 0.05) -> dict:
    data_dir = os.path.join(base_data_dir, f"{config_prefix}_seed{seed}")
    obs = v2.load_npz(os.path.join(data_dir, "obs.npz"))
    with open(os.path.join(data_dir, "oracle.pkl"), "rb") as f:
        oracle = pickle.load(f)
    cfg = v5.reload_cfg(oracle["config"])
    K, n, m = cfg.K, cfg.n_k, cfg.m_k
    k_holdout = K - 1

    rng = np.random.default_rng(seed * 31 + 17)
    rich = gen_rich_int_holdout(cfg, oracle["GV_edges"],
                                oracle["mechanism_params"], rng,
                                k_holdout, T_per_module=1000)
    rng2 = np.random.default_rng(seed * 31 + 19)
    ev = gen_unseen_int_eval(cfg, oracle["GV_edges"],
                             oracle["mechanism_params"], rng2,
                             k_holdout, T=1500)

    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval = ev["X"], ev["A"]

    pred_b10 = v2.b10_predict(X_eval, A_eval, oracle, K, n)
    pred_b2 = v2.b2_fit_predict(X_train, A_train, X_eval, A_eval, K, n, m)

    cand_obs = v2.b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i, j) in cand_obs:
        parents_b7[j].append(i)
    coefs_b7 = v2.fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)
    pred_b7 = v2.predict_per_module(X_eval, A_eval, coefs_b7, parents_b7, lag=1)

    icp = v5.icp_validate_all(rich["X"], rich["A"], rich["intervention_mask"],
                              K, n, m, alpha=alpha)
    parents_b8 = {k: [] for k in range(K)}
    for (i, j) in icp["validated_edges"]:
        parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)
    pred_b8 = v2.predict_per_module(X_eval, A_eval, coefs_b8, parents_b8, lag=1)

    metrics = {}
    for name, pred in [("B10", pred_b10), ("B2", pred_b2),
                       ("B7", pred_b7), ("B8", pred_b8)]:
        metrics[name] = {
            "mse": v5.cont_mse(pred, X_eval),
            "mse_per_module": v5.cont_mse_per_module(pred, X_eval),
            **v5.em_multi_bin(pred, X_eval, X_train, n_bins_list=(5, 20, 32)),
        }

    GM = np.asarray(oracle["GM_adj"])
    true_edges = [(int(i), int(j)) for i in range(K) for j in range(K) if GM[i, j] > 0]
    val_set = set((int(i), int(j)) for (i, j) in icp["validated_edges"])
    tp = len(val_set & set(true_edges))
    fp = len(val_set - set(true_edges))
    fn = len(set(true_edges) - val_set)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    th2 = v5.theorem2_trackers(rich["intervention_mask"], K)

    return {
        "seed": seed, "K": K, "k_holdout": k_holdout,
        "config_prefix": config_prefix,
        "true_edges": true_edges,
        "icp_validated_edges": icp["validated_edges"],
        "edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
        "alpha_icp": alpha,
        "metrics": metrics,
        "theorem2": th2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synthetic")
    ap.add_argument("--out_dir", default="runs/anchor_4_v6")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    runs = []
    plan = [
        ("sanity_chain_d4", 0.05),
        ("medium_chain_d4", 0.05),
    ]
    for config_prefix, alpha in plan:
        print(f"\n=== {config_prefix} / unseen_int eval / α={alpha} ===")
        per_seed = []
        for s in args.seeds:
            r = run_seed_v6(s, args.data_dir, config_prefix, alpha=alpha)
            per_seed.append(r)
            em32 = {n: r["metrics"][n]["em_32bin"] for n in ["B2", "B7", "B8", "B10"]}
            em20 = {n: r["metrics"][n]["em_20bin"] for n in ["B2", "B7", "B8", "B10"]}
            mse = {n: r["metrics"][n]["mse"] for n in ["B2", "B7", "B8", "B10"]}
            print(f"  seed {s} k_holdout={r['k_holdout']}")
            print(f"    EM_20bin B2={em20['B2']:.4f} B7={em20['B7']:.4f} "
                  f"B8={em20['B8']:.4f} B10={em20['B10']:.4f}  "
                  f"(B8-B2={(em20['B8']-em20['B2'])*100:.2f}pp)")
            print(f"    EM_32bin B2={em32['B2']:.4f} B7={em32['B7']:.4f} "
                  f"B8={em32['B8']:.4f} B10={em32['B10']:.4f}  "
                  f"(B8-B2={(em32['B8']-em32['B2'])*100:.2f}pp)")
            print(f"    MSE      B2={mse['B2']:.3f}  B7={mse['B7']:.3f}  "
                  f"B8={mse['B8']:.3f}  B10={mse['B10']:.3f}  "
                  f"(B8/B2 = {mse['B8']/mse['B2']:.3f})")
            print(f"    edge_f1=B8:{r['edge_f1']:.3f} (prec={r['edge_precision']:.2f} "
                  f"rec={r['edge_recall']:.2f})")
            print(f"    ICP validated: {r['icp_validated_edges']}")

        runs.append({"config_prefix": config_prefix, "per_seed": per_seed})

    summary = {"phase": "RUNNING_anchor_4_v6_unseen_int",
               "elapsed_seconds": time.time() - t0,
               "configs": []}
    for run in runs:
        per_seed = run["per_seed"]
        agg = {}
        for name in ["B2", "B7", "B8", "B10"]:
            agg[name] = {
                "mse": v5.aggregate(per_seed, ("metrics", name, "mse")),
                "em_5bin": v5.aggregate(per_seed, ("metrics", name, "em_5bin")),
                "em_20bin": v5.aggregate(per_seed, ("metrics", name, "em_20bin")),
                "em_32bin": v5.aggregate(per_seed, ("metrics", name, "em_32bin")),
            }
        edge_f1 = v5.aggregate(per_seed, ("edge_f1",))
        B8 = agg["B8"]["em_32bin"]["mean"]
        B10 = agg["B10"]["em_32bin"]["mean"]
        B2 = agg["B2"]["em_32bin"]["mean"]
        gate = {
            "primary_metric": "transition_em_32bin",
            "B10_em32_pp": round(B10 * 100, 2),
            "B8_em32_pp": round(B8 * 100, 2),
            "B7_em32_pp": round(agg["B7"]["em_32bin"]["mean"] * 100, 2),
            "B2_em32_pp": round(B2 * 100, 2),
            "B8_minus_B10_pp": round((B8 - B10) * 100, 2),
            "B8_minus_B2_pp": round((B8 - B2) * 100, 2),
            "criterion_1_pass": (B8 - B10) >= -0.03,
            "criterion_2_pass": (B8 - B2) >= 0.10,
            "edge_f1_B8": round(edge_f1["mean"], 3),
        }
        gate["overall_pass"] = gate["criterion_1_pass"] and gate["criterion_2_pass"]
        summary["configs"].append({
            "config_prefix": run["config_prefix"],
            "per_seed": per_seed,
            "aggregate": agg,
            "edge_f1_b8_mean": edge_f1["mean"],
            "gate": gate,
        })

    primary = summary["configs"][1]
    sanity = summary["configs"][0]
    summary["primary_gate"] = primary["gate"]
    summary["sanity_gate"] = sanity["gate"]

    out = os.path.join(args.out_dir, "anchor4_results_v6_unseen.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out}")
    print("=== PRIMARY GATE V6 (medium γ=0.5 + unseen-int eval) ===")
    print(json.dumps(summary["primary_gate"], indent=2))
    print("=== SANITY GATE V6 (sanity γ=0 + unseen-int eval) ===")
    print(json.dumps(summary["sanity_gate"], indent=2))
    return 0 if summary["primary_gate"]["overall_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
