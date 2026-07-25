"""Run V3 on medium_chain_d4 (gamma=0.5 — Theorem 1 bound active).

Also test on a confounder-shift eval split where δ_int² should drive B2 down.
"""

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
from experiments import anchor4_v3 as v3
from fed_causal.synthetic_scm_skeleton import simulate

base = "data/synthetic"
out_dir = "runs/anchor_4_v4"


def gen_confounder_eval(cfg, GV, params, rng, T=1500, conf_mu=2.5):
    """Eval under confounder-shift: U_conf ~ N(mu_shift, I) — Theorem 1 δ_int² bites here."""
    spec = {"confounder_shift": True, "conf_mu": conf_mu}
    return simulate(cfg, GV, params, spec, rng, T)


def run_seed_medium(seed, base_data_dir=base, n_bins=5):
    data_dir = os.path.join(base_data_dir, f"medium_chain_d4_seed{seed}")
    obs = v2.load_npz(os.path.join(data_dir, "obs.npz"))
    with open(os.path.join(data_dir, "oracle.pkl"), "rb") as f:
        oracle = pickle.load(f)
    cfg = v2.reload_cfg(oracle["config"])
    K, n, m = cfg.K, cfg.n_k, cfg.m_k

    rng = np.random.default_rng(seed*31+17)
    rich = v2.gen_rich_int_train(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                                  rng, T_per_module=800)
    # CONFOUNDER-SHIFT eval — Theorem 1's regime
    rng2 = np.random.default_rng(seed*31+19)
    eval_split = gen_confounder_eval(cfg, oracle["GV_edges"], oracle["mechanism_params"],
                                       rng2, T=1500, conf_mu=2.5)

    X_train, A_train = obs["X"], obs["A"]
    X_eval, A_eval = eval_split["X"], eval_split["A"]
    edges = v2.fit_bin_edges(X_train, n_bins=n_bins)
    true_disc = v2.discretize(X_eval, edges)

    pred_b10 = v2.b10_predict(X_eval, A_eval, oracle, K, n)
    em_b10 = v2.transition_em(v2.discretize(pred_b10, edges), true_disc)
    pred_b2 = v2.b2_fit_predict(X_train, A_train, X_eval, A_eval, K, n, m)
    em_b2 = v2.transition_em(v2.discretize(pred_b2, edges), true_disc)

    cand_obs = v2.b7_candidate_edges(X_train, K, threshold=0.10)
    parents_b7 = {k: [] for k in range(K)}
    for (i,j) in cand_obs: parents_b7[j].append(i)
    coefs_b7 = v2.fit_per_module_ridge(X_train, A_train, parents_b7, lag=1)
    pred_b7 = v2.predict_per_module(X_eval, A_eval, coefs_b7, parents_b7, lag=1)
    em_b7 = v2.transition_em(v2.discretize(pred_b7, edges), true_disc)

    matcher = v3.ate_step3_step4(rich["X"], rich["A"], rich["intervention_mask"], K, n,
                                 lag_window=1, p_verify_threshold=0.30)
    parents_b8 = {k: [] for k in range(K)}
    for (i,j) in matcher["validated_edges"]: parents_b8[j].append(i)
    coefs_b8 = v2.fit_per_module_ridge(X_train, A_train, parents_b8, lag=1)
    pred_b8 = v2.predict_per_module(X_eval, A_eval, coefs_b8, parents_b8, lag=1)
    em_b8 = v2.transition_em(v2.discretize(pred_b8, edges), true_disc)

    q_pos = [q for q in matcher["q_hat"] if q > 0]
    r_pos = [r for r in matcher["r_hat"] if r > 0]
    q_min = min(q_pos) if q_pos else 0.0
    r_min = min(r_pos) if r_pos else 0.0
    if q_min*r_min > 0:
        N_for_p95 = int(np.ceil(np.log(0.05) / np.log(1 - q_min*r_min)))
    else:
        N_for_p95 = float("inf")

    GM_adj = np.asarray(oracle["GM_adj"])
    true_edges = [(int(i),int(j)) for i in range(K) for j in range(K) if GM_adj[i,j] > 0]
    val_set = set((int(i),int(j)) for (i,j) in matcher["validated_edges"])
    true_set = set(true_edges)
    tp = len(val_set & true_set); fp = len(val_set - true_set); fn = len(true_set - val_set)
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    f1 = 2*prec*rec/max(1e-9, prec+rec)

    return {"seed":seed, "K":K, "true_edges":true_edges,
            "validated_edges":matcher["validated_edges"],
            "edge_precision":prec, "edge_recall":rec, "edge_f1":f1,
            "transition_em":{"B2":em_b2, "B7":em_b7, "B8":em_b8, "B10":em_b10},
            "per_module_em":{
                "B2":v2.per_module_em(v2.discretize(pred_b2, edges), true_disc),
                "B7":v2.per_module_em(v2.discretize(pred_b7, edges), true_disc),
                "B8":v2.per_module_em(v2.discretize(pred_b8, edges), true_disc),
                "B10":v2.per_module_em(v2.discretize(pred_b10, edges), true_disc),
            },
            "theorem2":{
                "q_hat":matcher["q_hat"], "r_hat":matcher["r_hat"],
                "q_min":q_min, "r_min":r_min,
                "N_min_validated":matcher["N_min_validated"],
                "N_for_p_verify_0_95":N_for_p95,
            }}


if __name__ == "__main__":
    t0 = time.time()
    per_seed = [run_seed_medium(s) for s in [0,1,2]]
    for r in per_seed:
        em = r["transition_em"]
        print(f"seed {r['seed']}: B2={em['B2']:.4f} B7={em['B7']:.4f} B8={em['B8']:.4f} B10={em['B10']:.4f}")
        print(f"  edge_f1={r['edge_f1']:.3f} (prec={r['edge_precision']:.2f} rec={r['edge_recall']:.2f})")
        print(f"  validated={r['validated_edges']}  true={r['true_edges']}")
        print(f"  N_min={r['theorem2']['N_min_validated']} N_for_p95={r['theorem2']['N_for_p_verify_0_95']}")

    def agg(m):
        vals = [s["transition_em"][m] for s in per_seed]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    B2, B7, B8, B10 = agg("B2"), agg("B7"), agg("B8"), agg("B10")
    print(f"\nAGG (medium gamma=0.5 + confounder-shift eval):")
    print(f"  B2={B2['mean']*100:.2f}±{B2['std']*100:.2f}pp")
    print(f"  B7={B7['mean']*100:.2f}±{B7['std']*100:.2f}pp")
    print(f"  B8={B8['mean']*100:.2f}±{B8['std']*100:.2f}pp")
    print(f"  B10={B10['mean']*100:.2f}±{B10['std']*100:.2f}pp")
    print(f"  B8-B10={(B8['mean']-B10['mean'])*100:.2f}pp (need >= -3)")
    print(f"  B8-B2={(B8['mean']-B2['mean'])*100:.2f}pp (need >= +10)")

    summary = {"phase":"RUNNING_anchor_4_v4_medium_confshift",
               "elapsed_seconds": time.time()-t0,
               "per_seed":per_seed,
               "aggregate":{"B2_em":B2,"B7_em":B7,"B8_em":B8,"B10_em":B10}}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anchor4_results_v4_medium.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"saved {out_path}")
