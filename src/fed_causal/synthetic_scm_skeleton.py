"""
synthetic_scm_skeleton.py — Modular SCM data generator for fedcausalworld.

Pure numpy / scipy / networkx. NO LLM calls. NO secrets. NO env reads.

Conforms to:
  - insight.md Assumption 1 (Modular SCM with independent U_k)
  - insight.md Assumption 2 (unblockable back-door path when gamma > 0)
  - modularization_spec.md schema (X_k disjoint, 6 modules, ground-truth edges)
  - synthetic_scm_design.md §5 (this is the executable companion)

Outputs (saved as .npz under data/synthetic/<config_id>/ by engineer):
  - X_obs, A_obs, X_int, A_int, intervention_mask
  - oracle_GM, oracle_GV (with lag), mechanism_params

Maintainer: anonymous artifact authors, 2026-06-19 JST
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse
import json
import os
import pickle

import numpy as np
import networkx as nx


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

@dataclass
class SCMConfig:
    K: int = 6
    n_k: int = 4
    m_k: int = 2
    rho: float = 0.30
    d: int = 4
    alpha: float = 0.50
    lag_mean: int = 1
    gamma: float = 0.30
    sigma: float = 0.10
    d_conf: int = 2
    pi_obs: float = 0.30
    T_train: int = 5000
    T_eval: int = 2000
    seed: int = 0
    # EDA A3 fix: force_chain enforces chain 0->1->...->chain_depth
    # so measured path depth equals chain_depth (random DAG with rho=0.3
    # only gives depth ~2, see eda_report.md §6.3).
    force_chain: bool = False
    chain_depth: int = 4
    chain_side_prob: float = 0.0
    # anchor_4_scm_fix: --confound_action makes U_conf influence BOTH X and A,
    # creating a real backdoor X <- U_conf -> A -> X required by
    # hypothesis.md Assumption 2. Without this, P(X[t+1]|X[t],A[t]) is
    # identifiable from obs alone, so non-causal OLS matches oracle.
    # Policy: A[t,k,q] ~ Bernoulli(sigmoid(logit(pi_obs) + gamma_A * (V_A_k @ U_conf[t])[q]))
    confound_action: bool = False
    gamma_A: float = 0.0     # strength of U_conf -> A coupling (0 = original behavior)


# -----------------------------------------------------------------------------
# 2. Graph construction
# -----------------------------------------------------------------------------

def build_module_graph(cfg: SCMConfig, rng: np.random.Generator) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(cfg.K))

    # NEW (EDA A3 fix): --force_chain mode generates a deterministic chain
    # 0 -> 1 -> 2 -> ... -> chain_depth, ensuring measured path depth = chain_depth.
    # This is required for Exp6 depth-scan since random DAGs with rho=0.3 yield
    # depth ~2 even when cfg.d=4 (see eda_report.md A3).
    if getattr(cfg, "force_chain", False):
        cd = getattr(cfg, "chain_depth", cfg.d)
        cd = max(1, min(cd, cfg.K - 1))
        for i in range(cd):
            G.add_edge(i, i + 1)
        # add a sparse random side-edge (i -> i+2) with low prob to enrich
        # graph beyond pure chain, but only if it does not exceed cd path-len.
        side_p = getattr(cfg, "chain_side_prob", 0.0)
        if side_p > 0.0:
            for i in range(cd - 1):
                if rng.random() < side_p:
                    G.add_edge(i, i + 2)
            # Trim if side-edges accidentally lengthen path (shouldn't, but safe)
            while nx.is_directed_acyclic_graph(G) \
                    and nx.dag_longest_path_length(G) > cd:
                bet = nx.edge_betweenness_centrality(G)
                # prefer removing side edges, not chain edges
                side_edges = [e for e in G.edges if e[1] - e[0] != 1]
                if side_edges:
                    worst = max(side_edges, key=lambda e: bet.get(e, 0))
                else:
                    worst = max(bet, key=bet.get)
                G.remove_edge(*worst)
        return G

    for i in range(cfg.K):
        for j in range(i + 1, cfg.K):
            if rng.random() < cfg.rho:
                G.add_edge(i, j)

    # Trim edges to enforce max longest-path length <= d
    while len(G.edges) > 0 and nx.is_directed_acyclic_graph(G) \
            and nx.dag_longest_path_length(G) > cfg.d:
        bet = nx.edge_betweenness_centrality(G)
        worst = max(bet, key=bet.get)
        G.remove_edge(*worst)

    # Ensure at least one cross-module edge (Assumption 1: |E_M^*| >= 1)
    if len(G.edges) == 0:
        G.add_edge(0, 1)
    return G


def build_variable_edges(cfg: SCMConfig, GM: nx.DiGraph,
                         rng: np.random.Generator) -> List[Tuple[int, int, int, int, int]]:
    edges = []
    L = max(1, 2 * cfg.lag_mean)
    for (i, j) in GM.edges:
        p = int(rng.integers(0, cfg.n_k))
        q = int(rng.integers(0, cfg.n_k))
        lag = int(rng.integers(0, L + 1))
        edges.append((i, p, j, q, lag))
    return edges


# -----------------------------------------------------------------------------
# 3. Mechanism parameter sampling
# -----------------------------------------------------------------------------

def sample_mechanism_params(cfg: SCMConfig, GM: nx.DiGraph,
                            GV_edges: List[Tuple[int, int, int, int, int]],
                            rng: np.random.Generator) -> Dict:
    params = {"W_self": [], "B": [], "W_cross": {}, "V": []}

    for k in range(cfg.K):
        W = rng.normal(0, 0.1, size=(cfg.n_k, cfg.n_k))
        eig = np.max(np.abs(np.linalg.eigvals(W)))
        if eig > 0:
            W *= 0.9 / max(eig, 1e-6)
        params["W_self"].append(W)
        params["B"].append(rng.normal(0, 0.5, size=(cfg.n_k, cfg.m_k)))

    for (i, p, j, q, lag) in GV_edges:
        W = np.zeros((cfg.n_k, cfg.n_k))
        W[q, p] = rng.normal(1.0, 0.2)
        params["W_cross"][(i, j, lag)] = W

    chosen = set(rng.choice(cfg.K, size=max(2, cfg.K // 2), replace=False).tolist())
    for k in range(cfg.K):
        if k in chosen:
            params["V"].append(rng.normal(0, 1.0, size=(cfg.n_k, cfg.d_conf)))
        else:
            params["V"].append(np.zeros((cfg.n_k, cfg.d_conf)))

    # V_A: action-confounder loading per module. Same modules selected by
    # `chosen` get nonzero V_A (mirrors V structure so confounder enters both
    # X and A through SAME modules -> creates X <- U_conf -> A backdoor).
    params["V_A"] = []
    for k in range(cfg.K):
        if k in chosen:
            params["V_A"].append(rng.normal(0, 1.0, size=(cfg.m_k, cfg.d_conf)))
        else:
            params["V_A"].append(np.zeros((cfg.m_k, cfg.d_conf)))

    return params


# -----------------------------------------------------------------------------
# 4. Trajectory simulation
# -----------------------------------------------------------------------------

def simulate(cfg: SCMConfig, GV_edges, params,
             intervention_spec: Dict, rng: np.random.Generator,
             T: int) -> Dict[str, np.ndarray]:
    K, n, m = cfg.K, cfg.n_k, cfg.m_k
    L = max(1, 2 * cfg.lag_mean)
    X = np.zeros((T + L, K, n))
    A = np.zeros((T + L, K, m))
    int_mask = np.zeros((T, K), dtype=np.int8)

    pi = cfg.pi_obs
    if intervention_spec.get("policy_shift"):
        pi = 1.0 - cfg.pi_obs

    if intervention_spec.get("confounder_shift"):
        mu = intervention_spec.get("conf_mu", 2.0)
        U_conf = rng.normal(mu, 1.0, size=(T + L, cfg.d_conf))
    else:
        U_conf = rng.normal(0, 1.0, size=(T + L, cfg.d_conf))

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0)))

    confound_action = bool(getattr(cfg, "confound_action", False))
    gamma_A = float(getattr(cfg, "gamma_A", 0.0))
    if confound_action and gamma_A > 0:
        # logit(pi) is the base; add per-module per-action-dim shift from V_A @ U_conf
        eps = 1e-6
        pi_clipped = float(np.clip(pi, eps, 1 - eps))
        logit_pi = np.log(pi_clipped / (1.0 - pi_clipped))

    for t in range(L, T + L):
        # Sample actions
        if confound_action and gamma_A > 0:
            # A[t, k, q] ~ Bernoulli(sigmoid(logit_pi + gamma_A * (V_A_k @ U_conf[t])[q]))
            probs = np.empty((K, m))
            for k in range(K):
                shift = params["V_A"][k] @ U_conf[t]   # (m,)
                probs[k] = _sigmoid(logit_pi + gamma_A * shift)
            A[t] = (rng.random(size=(K, m)) < probs).astype(float)
        else:
            A[t] = (rng.random(size=(K, m)) < pi).astype(float)

        # Module-level do() injection
        if "do_module" in intervention_spec \
                and rng.random() < intervention_spec.get("rate", 0.10):
            k_do = intervention_spec["do_module"]
            A[t, k_do] = intervention_spec["do_value"]
            int_mask[t - L, k_do] = 1

        # Update each module
        for k in range(K):
            x_next = params["W_self"][k] @ X[t - 1, k] + params["B"][k] @ A[t, k]

            for (i, p, j, q, lag) in GV_edges:
                if j != k:
                    continue
                src = X[t - 1 - lag, i]
                signal = params["W_cross"][(i, j, lag)] @ np.tanh(src)
                x_next += cfg.alpha * signal

            x_next += cfg.gamma * params["V"][k] @ U_conf[t]
            x_next += rng.normal(0, cfg.sigma, size=n)

            if "do_var" in intervention_spec \
                    and intervention_spec["do_var"][0] == k \
                    and rng.random() < intervention_spec.get("rate", 0.10):
                _, q_do = intervention_spec["do_var"]
                x_next[q_do] = intervention_spec["do_value"]
                int_mask[t - L, k] = 1

            X[t, k] = x_next

    return {
        "X": X[L:],
        "A": A[L:],
        "intervention_mask": int_mask,
        "U_conf": U_conf[L:],
    }


# -----------------------------------------------------------------------------
# 5. Split generation
# -----------------------------------------------------------------------------

def generate_all_splits(cfg: SCMConfig) -> Dict[str, Dict]:
    rng = np.random.default_rng(cfg.seed)
    GM = build_module_graph(cfg, rng)
    GV_edges = build_variable_edges(cfg, GM, rng)
    params = sample_mechanism_params(cfg, GM, GV_edges, rng)

    in_out = {k: GM.in_degree(k) + GM.out_degree(k) for k in GM.nodes}
    k_mediator = max(in_out, key=in_out.get)
    k_holdout = max(GM.nodes)

    splits = {}
    splits["obs"] = simulate(cfg, GV_edges, params, {}, rng, cfg.T_train + cfg.T_eval)
    splits["policy_shift_train"] = simulate(cfg, GV_edges, params, {}, rng, cfg.T_train)
    splits["policy_shift_test"] = simulate(cfg, GV_edges, params,
                                           {"policy_shift": True}, rng, cfg.T_eval)
    spec_seen = {"do_module": 0, "do_value": np.ones(cfg.m_k), "rate": 0.10}
    spec_unseen = {"do_module": k_holdout, "do_value": np.ones(cfg.m_k), "rate": 0.20}
    splits["unseen_int_train"] = simulate(cfg, GV_edges, params, spec_seen, rng, cfg.T_train)
    splits["unseen_int_test"] = simulate(cfg, GV_edges, params, spec_unseen, rng, cfg.T_eval)
    spec_med = {"do_var": (k_mediator, 0), "do_value": 1.5, "rate": 0.30}
    splits["mediator_int"] = simulate(cfg, GV_edges, params, spec_med, rng, cfg.T_eval)
    splits["confounder_shift"] = simulate(cfg, GV_edges, params,
                                          {"confounder_shift": True, "conf_mu": 2.0},
                                          rng, cfg.T_eval)

    oracle = {
        "GM_adj": nx.to_numpy_array(GM, nodelist=list(range(cfg.K))),
        "GV_edges": GV_edges,
        "k_mediator": k_mediator,
        "k_holdout": k_holdout,
        "mechanism_params": params,
        "config": cfg.__dict__,
    }
    return {"splits": splits, "oracle": oracle, "GM_edges": list(GM.edges)}


# -----------------------------------------------------------------------------
# 6. Sanity checks (run before saving)
# -----------------------------------------------------------------------------

def sanity_check(cfg: SCMConfig, out: Dict) -> Dict:
    GM_adj = out["oracle"]["GM_adj"]
    K = cfg.K
    checks = {}

    # 1. DAG
    G = nx.from_numpy_array(GM_adj, create_using=nx.DiGraph)
    checks["is_dag"] = bool(nx.is_directed_acyclic_graph(G))

    # 2. At least one cross-module edge (Assumption 1)
    checks["has_cross_module_edge"] = bool(GM_adj.sum() >= 1)

    # 3. Longest path within budget
    if checks["is_dag"] and GM_adj.sum() > 0:
        checks["max_path_len"] = int(nx.dag_longest_path_length(G))
        checks["depth_within_budget"] = checks["max_path_len"] <= cfg.d
    else:
        checks["max_path_len"] = 0
        checks["depth_within_budget"] = True

    # 4. At least 2 modules have nonzero V_k when gamma > 0 (Assumption 2)
    nonzero_V = sum(1 for V in out["oracle"]["mechanism_params"]["V"]
                    if np.linalg.norm(V) > 0)
    checks["assumption_2_ok"] = (cfg.gamma == 0.0) or (nonzero_V >= 2)
    checks["nonzero_V_count"] = nonzero_V

    # 5. Local noise independence (empirical residual check on obs split)
    X = out["splits"]["obs"]["X"]
    # Residual = X[t+1] - W_self @ X[t] (approx) — use raw diff for quick check
    dX = X[1:] - X[:-1]
    flat = dX.reshape(dX.shape[0], -1)
    if flat.shape[1] >= 2:
        C = np.corrcoef(flat.T)
        np.fill_diagonal(C, 0.0)
        # Group by module
        max_inter_mod = 0.0
        for ki in range(K):
            for kj in range(K):
                if ki == kj:
                    continue
                block = C[ki*cfg.n_k:(ki+1)*cfg.n_k, kj*cfg.n_k:(kj+1)*cfg.n_k]
                max_inter_mod = max(max_inter_mod, float(np.max(np.abs(block))))
        # Inter-module correlation reflects the true cross-module signals,
        # so we don't require it to be small. Just report.
        checks["max_inter_module_corr"] = max_inter_mod

    return checks


# -----------------------------------------------------------------------------
# 7. CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic Modular SCM generator")
    parser.add_argument("--config_id", type=str, required=True)
    parser.add_argument("--out_dir", type=str,
                        default="/home/user/fedcausalworld/data/synthetic")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--n_k", type=int, default=4)
    parser.add_argument("--m_k", type=int, default=2)
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--lag_mean", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.30)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--T_train", type=int, default=5000)
    parser.add_argument("--T_eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    # EDA A3: force_chain mode for Exp6 depth scan
    parser.add_argument("--force_chain", action="store_true",
                        help="Force chain DAG 0->1->...->chain_depth; required for Exp6 depth scan (EDA A3)")
    parser.add_argument("--chain_depth", type=int, default=4,
                        help="Chain length when --force_chain (path depth = chain_depth)")
    parser.add_argument("--chain_side_prob", type=float, default=0.0,
                        help="Optional probability of side-edge (i -> i+2) when --force_chain")
    # anchor_4_scm_fix: confound_action makes U_conf influence both X and A
    parser.add_argument("--confound_action", action="store_true",
                        help="Backdoor: U_conf -> A as well as U_conf -> X (hypothesis.md Assumption 2 full)")
    parser.add_argument("--gamma_A", type=float, default=0.0,
                        help="Strength of U_conf -> A coupling (0 = original behavior)")
    args = parser.parse_args()

    cfg = SCMConfig(
        K=args.K, n_k=args.n_k, m_k=args.m_k, rho=args.rho, d=args.d,
        alpha=args.alpha, lag_mean=args.lag_mean, gamma=args.gamma,
        sigma=args.sigma, T_train=args.T_train, T_eval=args.T_eval, seed=args.seed,
        force_chain=args.force_chain, chain_depth=args.chain_depth,
        chain_side_prob=args.chain_side_prob,
        confound_action=args.confound_action, gamma_A=args.gamma_A,
    )

    out = generate_all_splits(cfg)
    checks = sanity_check(cfg, out)

    save_dir = os.path.join(args.out_dir, args.config_id)
    os.makedirs(save_dir, exist_ok=True)

    for name, data in out["splits"].items():
        np.savez_compressed(os.path.join(save_dir, f"{name}.npz"), **data)

    with open(os.path.join(save_dir, "oracle.pkl"), "wb") as f:
        pickle.dump(out["oracle"], f)

    with open(os.path.join(save_dir, "sanity.json"), "w") as f:
        json.dump(checks, f, indent=2, default=str)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"[OK] config_id={args.config_id} saved to {save_dir}")
    print(f"[CHECKS] {json.dumps(checks, indent=2, default=str)}")


if __name__ == "__main__":
    main()
