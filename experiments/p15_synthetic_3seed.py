import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def gen_and_eval(K, chain_depth, alpha, gamma, gamma_A, lag_mean=0,
                 sigma=0.10, T_train=3000, T_eval=1500, seed=0):
    from fed_causal.synthetic_scm_skeleton import (
        SCMConfig,
        generate_all_splits,
        simulate,
    )
    from experiments import anchor4_sanity_v2 as v2
    from experiments import anchor4_v5_icp as v5
    from experiments import anchor4_v9_b10fix as v9

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
    ap.add_argument("--out_dir", default="runs/p15_synthetic_3seed")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()
    out_path = args.out_path or os.path.join(
        args.out_dir, "p1_synthetic_3seed_summary.json"
    )

    t0 = time.time()
    print("=== P1.5: 3-seed Synthetic G6 (γ, α, d sweep) ===")

    d_values = [1, 2, 3, 4, 6, 8]
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]

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

    gamma_agg = []
    for i, g in enumerate(gamma_values):
        ds = [g_runs_per_seed[s][i]['delta_causal_MSE'] for s in args.seeds]
        gamma_agg.append({"gamma": g, "mean": float(np.mean(ds)),
                          "std": float(np.std(ds))})
    gamma_means = [a['mean'] for a in gamma_agg]
    rho_g = spearman_rho(gamma_values, gamma_means)
    print(f"\nγ: aggregate means {[round(m, 3) for m in gamma_means]}, ρ={rho_g:+.3f}")

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
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] saved {out_path}, elapsed {elapsed:.0f}s")
    print(f"G6 verdict: ρ_γ={rho_g:+.3f}, ρ_α={rho_a:+.3f}, ρ_d={rho_d:+.3f}")
    print(f"G6 n_pass (≥0.7): {summary['G6_n_pass']}/3")


def diagnostics_make_rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, 104729, stream]))


def diagnostics_average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def diagnostics_spearman(xs, ys):
    if len(xs) < 2:
        return None
    x = diagnostics_average_ranks(xs)
    y = diagnostics_average_ranks(ys)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else None


def diagnostics_fit_linear(features, targets):
    design = np.column_stack([np.ones(len(features)), features])
    return np.linalg.lstsq(design, targets, rcond=None)[0]


def diagnostics_predict_linear(features, coefficients):
    return np.column_stack([np.ones(len(features)), features]) @ coefficients


def diagnostics_sample_chain(args, seed, stream, size, depth, gamma, proxy_strength, regime):
    rng = diagnostics_make_rng(seed, stream)
    latent = rng.normal(size=size)
    proxy = proxy_strength * latent + rng.normal(size=size)
    action_noise = rng.normal(size=size)
    if regime == "observational":
        action = args.action_confounding * latent + action_noise
    elif regime == "randomized_training":
        action = action_noise * args.randomized_action_scale
    elif regime == "fixed_intervention":
        action = np.full(size, args.intervention_value)
    else:
        raise ValueError(regime)
    innovations = rng.normal(
        scale=args.outcome_noise,
        size=(size, max(args.depth_values) + 1),
    )[:, :depth + 1]
    outcomes = np.empty((size, depth + 1))
    outcomes[:, 0] = args.action_effect * action + gamma * latent + innovations[:, 0]
    for node in range(1, depth + 1):
        outcomes[:, node] = args.chain_coupling * outcomes[:, node - 1] + innovations[:, node]
    return np.column_stack([action, proxy]), outcomes


def diagnostics_chain_record(args, seed, sweep, axis, value):
    gamma = value if sweep == "confounding" else args.confounding
    proxy_strength = value if sweep == "proxy" else args.proxy_strength
    depth = int(value) if sweep == "depth" else 0
    features_obs, outcomes_obs = diagnostics_sample_chain(
        args, seed, 1, args.n_observational, depth, gamma, proxy_strength, "observational"
    )
    features_int, outcomes_int = diagnostics_sample_chain(
        args, seed, 2, args.n_interventional, depth, gamma, proxy_strength, "randomized_training"
    )
    features_eval, outcomes_eval = diagnostics_sample_chain(
        args, seed, 3, args.n_eval, depth, gamma, proxy_strength, "fixed_intervention"
    )
    observational_fit = diagnostics_fit_linear(features_obs, outcomes_obs)
    observational_prediction = diagnostics_predict_linear(features_eval, observational_fit)
    root_fit = diagnostics_fit_linear(features_int, outcomes_int[:, 0])
    causal_prediction = np.empty_like(outcomes_eval)
    causal_prediction[:, 0] = diagnostics_predict_linear(features_eval, root_fit)
    local_fits = []
    for node in range(1, depth + 1):
        coefficients = diagnostics_fit_linear(outcomes_int[:, node - 1:node], outcomes_int[:, node])
        local_fits.append(coefficients)
        causal_prediction[:, node] = diagnostics_predict_linear(causal_prediction[:, node - 1:node], coefficients)
    action, proxy = features_eval.T
    latent_mean_int = proxy_strength * proxy / (1.0 + proxy_strength ** 2)
    latent_mean_obs = (
        args.action_confounding * action + proxy_strength * proxy
    ) / (1.0 + args.action_confounding ** 2 + proxy_strength ** 2)
    powers = args.chain_coupling ** np.arange(depth + 1)
    oracle = (args.action_effect * action + gamma * latent_mean_int)[:, None] * powers
    population_observational = (args.action_effect * action + gamma * latent_mean_obs)[:, None] * powers
    predictors = {
        "fitted_observational": observational_prediction,
        "population_observational": population_observational,
        "learned_causal": causal_prediction,
        "interventional_conditional_mean": oracle,
    }
    node_mse = {
        name: np.mean((prediction - outcomes_eval) ** 2, axis=0)
        for name, prediction in predictors.items()
    }
    metrics = {}
    for name, errors in node_mse.items():
        metrics[name + "_terminal_mse"] = float(errors[-1])
        metrics[name + "_all_node_mse"] = float(errors.mean())
    oracle_risk = metrics["interventional_conditional_mean_terminal_mse"]
    metrics["fitted_observational_minus_oracle"] = metrics["fitted_observational_terminal_mse"] - oracle_risk
    metrics["population_observational_minus_oracle"] = metrics["population_observational_terminal_mse"] - oracle_risk
    metrics["fitted_observational_minus_learned_causal"] = (
        metrics["fitted_observational_terminal_mse"] - metrics["learned_causal_terminal_mse"]
    )
    metrics["population_conditional_mean_gap_squared"] = float(
        np.mean((population_observational[:, -1] - oracle[:, -1]) ** 2)
    )
    metrics["finite_sample_risk_decomposition_residual"] = (
        metrics["population_observational_minus_oracle"] - metrics["population_conditional_mean_gap_squared"]
    )
    denominator = 1.0 + args.action_confounding ** 2 + proxy_strength ** 2
    analytic_root_gap = gamma ** 2 * args.action_confounding ** 2 / denominator ** 2 * (
        args.intervention_value ** 2
        + proxy_strength ** 2 * args.action_confounding ** 2 / (1.0 + proxy_strength ** 2)
    )
    metrics["analytic_population_excess_risk"] = float(powers[-1] ** 2 * analytic_root_gap)
    metrics["analytic_oracle_risk"] = float(
        powers[-1] ** 2 * gamma ** 2 / (1.0 + proxy_strength ** 2)
        + args.outcome_noise ** 2 * np.sum(powers ** 2)
    )
    local_errors = np.empty_like(causal_prediction)
    local_errors[:, 0] = causal_prediction[:, 0] - oracle[:, 0]
    for node, coefficients in enumerate(local_fits, start=1):
        local_errors[:, node] = coefficients[0] + (
            coefficients[1] - args.chain_coupling
        ) * causal_prediction[:, node - 1]
    local_mse = np.mean(local_errors ** 2, axis=0)
    accumulated_error = local_errors @ powers[::-1]
    metrics["learned_causal_conditional_mean_error"] = float(np.mean((causal_prediction[:, -1] - oracle[:, -1]) ** 2))
    metrics["maximum_local_mechanism_mse"] = float(local_mse.max())
    metrics["causal_error_path_bound"] = float(local_mse.max() * np.sum(np.abs(powers)) ** 2)
    metrics["path_error_identity_max_residual"] = float(
        np.max(np.abs(accumulated_error - causal_prediction[:, -1] + oracle[:, -1]))
    )
    return {
        "sweep": sweep,
        "sweep_axis": axis,
        "seed": seed,
        "condition": {axis: value},
        "config": {
            "depth": depth,
            "outcome_confounding": gamma,
            "observed_proxy_strength": proxy_strength,
            "action_confounding": args.action_confounding,
            "action_effect": args.action_effect,
            "chain_coupling": args.chain_coupling,
            "outcome_noise": args.outcome_noise,
            "intervention_value": args.intervention_value,
            "n_observational": args.n_observational,
            "n_interventional": args.n_interventional,
            "n_eval": args.n_eval,
            "rng_streams": {"observational": 1, "randomized_training": 2, "fixed_intervention": 3},
        },
        "metrics": metrics,
        "per_node_mse": {name: values.tolist() for name, values in node_mse.items()},
        "local_mechanism_mse": local_mse.tolist(),
        "fitted_coefficients": {
            "observational": observational_fit.tolist(),
            "causal_root": root_fit.tolist(),
            "causal_edges": [values.tolist() for values in local_fits],
        },
    }


def diagnostics_coverage_records(args, seed):
    records = []
    exposures = diagnostics_make_rng(seed, 11).random((args.coverage_repeats, max(args.coverage_budgets), args.true_edges))
    responses = diagnostics_make_rng(seed, 12).random(exposures.shape)
    matches = (exposures < args.exposure_probability) & (responses < args.response_probability)
    false_matches = diagnostics_make_rng(seed, 13).random(
        (args.coverage_repeats, max(args.coverage_budgets), args.non_edges)
    ) < args.exposure_probability * args.false_match_probability
    for budget in args.coverage_budgets:
        true_positive = matches[:, :budget].any(axis=1).sum(axis=1)
        false_positive = false_matches[:, :budget].any(axis=1).sum(axis=1)
        denominator = true_positive + false_positive
        precision = np.divide(true_positive, denominator, out=np.zeros(args.coverage_repeats), where=denominator > 0)
        recall = true_positive / args.true_edges
        f1 = 2.0 * true_positive / (args.true_edges + true_positive + false_positive)
        q_r = args.exposure_probability * args.response_probability
        miss_edge = (1.0 - q_r) ** budget
        metrics = {
            "true_edge_recall": float(recall.mean()),
            "precision_zero_when_no_prediction": float(precision.mean()),
            "edge_f1": float(f1.mean()),
            "all_true_edges_covered_probability": float(np.mean(true_positive == args.true_edges)),
            "any_true_edge_missing_probability": float(np.mean(true_positive < args.true_edges)),
            "expected_true_edge_recall": 1.0 - miss_edge,
            "exact_all_true_edges_covered_probability": (1.0 - miss_edge) ** args.true_edges,
            "exact_any_true_edge_missing_probability": 1.0 - (1.0 - miss_edge) ** args.true_edges,
            "missing_edge_union_bound": min(1.0, args.true_edges * miss_edge),
            "missing_edge_exponential_union_bound": min(1.0, args.true_edges * float(np.exp(-budget * q_r))),
            "mean_false_positive_edges": float(false_positive.mean()),
        }
        records.append({
            "sweep": "coverage",
            "sweep_axis": "verification_budget",
            "seed": seed,
            "condition": {"verification_budget": budget},
            "config": {
                "true_edges": args.true_edges,
                "non_edges": args.non_edges,
                "exposure_probability": args.exposure_probability,
                "response_probability": args.response_probability,
                "false_match_probability": args.false_match_probability,
                "repeats": args.coverage_repeats,
                "independent_edges_and_trials": True,
                "recovery_rule": "at_least_one_observed_match",
                "rng_streams": {"exposure": 11, "response": 12, "false_matches": 13},
            },
            "metrics": metrics,
            "per_repeat": {"true_positive": true_positive.tolist(), "false_positive": false_positive.tolist()},
        })
    return records


def diagnostics_spectral_records(args, seed):
    records = []
    matrix_rng = diagnostics_make_rng(seed, 21)
    basis, _ = np.linalg.qr(matrix_rng.normal(size=(args.spectral_dimension, args.spectral_dimension)))
    errors = diagnostics_make_rng(seed, 22).normal(scale=args.mechanism_error, size=(args.spectral_samples, args.spectral_dimension))
    error_energy = float(np.mean(np.sum(errors ** 2, axis=1)))
    identity = np.eye(args.spectral_dimension)
    for radius in args.spectral_radii:
        eigenvalues = radius * np.linspace(1.0, 0.25, args.spectral_dimension)
        matrix = (basis * eigenvalues) @ basis.T
        propagation = identity.copy()
        power = identity.copy()
        for horizon in range(max(args.spectral_horizons) + 1):
            if horizon > 0:
                power = power @ matrix
                propagation += power
            if horizon not in args.spectral_horizons:
                continue
            propagated = errors @ propagation.T
            gain = float(np.linalg.norm(propagation, ord=2))
            metrics = {
                "spectral_radius": float(np.max(np.abs(np.linalg.eigvalsh(matrix)))),
                "operator_norm": gain,
                "squared_operator_norm": gain ** 2,
                "local_error_energy": error_energy,
                "propagated_error_energy": float(np.mean(np.sum(propagated ** 2, axis=1))),
                "analytic_propagated_error_energy": float(args.mechanism_error ** 2 * np.sum(propagation ** 2)),
                "empirical_operator_error_bound": gain ** 2 * error_energy,
                "population_operator_error_bound": gain ** 2 * args.spectral_dimension * args.mechanism_error ** 2,
                "resolvent_identity_residual": float(np.linalg.norm((identity - matrix) @ propagation - identity + power @ matrix)),
            }
            if radius < 1.0:
                resolvent = np.linalg.solve(identity - matrix, identity)
                metrics["distance_to_stable_resolvent"] = float(np.linalg.norm(propagation - resolvent, ord=2))
                metrics["infinite_horizon_squared_operator_norm"] = float(np.linalg.norm(resolvent, ord=2) ** 2)
                metrics["normal_spectral_bound_squared"] = float(1.0 / np.min(np.abs(1.0 - eigenvalues)) ** 2)
            records.append({
                "sweep": "spectral",
                "sweep_axis": "spectral_radius",
                "seed": seed,
                "condition": {"spectral_radius": radius, "horizon": horizon},
                "config": {
                    "dimension": args.spectral_dimension,
                    "samples": args.spectral_samples,
                    "mechanism_error_std": args.mechanism_error,
                    "matrix_family": "real_symmetric_nonnegative_eigenvalues",
                    "propagation": "sum_B_power_ell_for_ell_0_through_horizon",
                    "error_model": "one_fixed_local_error_vector_repeated_at_each_composition",
                    "regime": "stable" if radius < 1.0 else "critical" if radius == 1.0 else "unstable",
                    "rng_streams": {"matrix": 21, "errors": 22},
                },
                "metrics": metrics,
                "eigenvalues": eigenvalues.tolist(),
                "propagation_matrix": matrix.tolist(),
            })
    return records


def diagnostics_summarize(records):
    grouped = defaultdict(list)
    for record in records:
        key = (record["sweep"], json.dumps(record["condition"], sort_keys=True))
        grouped[key].append(record)
    aggregates = []
    for group in grouped.values():
        metrics = {}
        names = sorted(set.intersection(*(set(record["metrics"]) for record in group)))
        for name in names:
            values = [record["metrics"][name] for record in group]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "n": len(values),
            }
        aggregates.append({
            "sweep": group[0]["sweep"],
            "sweep_axis": group[0]["sweep_axis"],
            "condition": group[0]["condition"],
            "metrics": metrics,
        })
    trend_groups = defaultdict(list)
    for record in aggregates:
        fixed = {key: value for key, value in record["condition"].items() if key != record["sweep_axis"]}
        trend_groups[(record["sweep"], record["sweep_axis"], json.dumps(fixed, sort_keys=True))].append(record)
        if record["sweep"] == "spectral":
            fixed = {"spectral_radius": record["condition"]["spectral_radius"]}
            trend_groups[("spectral", "horizon", json.dumps(fixed, sort_keys=True))].append(record)
    correlations = []
    for (sweep, axis, fixed), group in trend_groups.items():
        names = sorted(set.intersection(*(set(record["metrics"]) for record in group)))
        correlations.append({
            "sweep": sweep,
            "axis": axis,
            "fixed_condition": json.loads(fixed),
            "n_conditions": len(group),
            "spearman_over_seed_means": {
                name: diagnostics_spearman(
                    [record["condition"][axis] for record in group],
                    [record["metrics"][name]["mean"] for record in group],
                ) for name in names
            },
        })
    return aggregates, correlations


def diagnostics_parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory-sweeps", action="store_true")
    parser.add_argument("--sweeps", nargs="+", choices=["confounding", "proxy", "depth", "coverage", "spectral"], default=["confounding", "proxy", "depth", "coverage", "spectral"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", "--out_dir", default="runs/p15_theory_diagnostics")
    parser.add_argument("--out-path", "--out_path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-observational", type=int, default=3000)
    parser.add_argument("--n-interventional", type=int, default=3000)
    parser.add_argument("--n-eval", type=int, default=1500)
    parser.add_argument("--confounding-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--proxy-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--depth-values", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--confounding", type=float, default=1.0)
    parser.add_argument("--proxy-strength", type=float, default=0.0)
    parser.add_argument("--action-confounding", type=float, default=1.0)
    parser.add_argument("--action-effect", type=float, default=1.0)
    parser.add_argument("--chain-coupling", type=float, default=1.1)
    parser.add_argument("--outcome-noise", type=float, default=0.2)
    parser.add_argument("--intervention-value", type=float, default=2.0)
    parser.add_argument("--randomized-action-scale", type=float, default=1.0)
    parser.add_argument("--coverage-budgets", nargs="+", type=int, default=[0, 1, 5, 10, 25, 50, 100])
    parser.add_argument("--coverage-repeats", type=int, default=1000)
    parser.add_argument("--true-edges", type=int, default=8)
    parser.add_argument("--non-edges", type=int, default=20)
    parser.add_argument("--exposure-probability", type=float, default=0.2)
    parser.add_argument("--response-probability", type=float, default=0.5)
    parser.add_argument("--false-match-probability", type=float, default=0.0)
    parser.add_argument("--spectral-radii", nargs="+", type=float, default=[0.25, 0.5, 0.8, 0.95, 1.0, 1.05])
    parser.add_argument("--spectral-horizons", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--spectral-dimension", type=int, default=8)
    parser.add_argument("--spectral-samples", type=int, default=1500)
    parser.add_argument("--mechanism-error", type=float, default=0.05)
    args = parser.parse_args(argv)
    if len(args.sweeps) != len(set(args.sweeps)):
        parser.error("sweeps must contain unique values")
    for name in ("n_observational", "n_interventional", "n_eval", "coverage_repeats", "true_edges", "spectral_dimension", "spectral_samples"):
        if getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    if min(args.n_observational, args.n_interventional) < 4:
        parser.error("training splits must contain at least four samples")
    for name in ("seeds", "depth_values", "coverage_budgets", "spectral_horizons", "spectral_radii", "confounding_values", "proxy_values"):
        values = getattr(args, name)
        if any(not np.isfinite(value) or value < 0 for value in values):
            parser.error(name + " must contain finite nonnegative values")
        if len(values) != len(set(values)):
            parser.error(name + " must contain unique values")
    for name in ("exposure_probability", "response_probability", "false_match_probability"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(name + " must be in [0, 1]")
    for name in ("outcome_noise", "mechanism_error", "proxy_strength", "confounding"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(name + " must be finite and nonnegative")
    for name in ("action_confounding", "action_effect", "chain_coupling", "intervention_value", "randomized_action_scale"):
        if not np.isfinite(getattr(args, name)):
            parser.error(name + " must be finite")
    if args.randomized_action_scale <= 0 or args.non_edges < 0:
        parser.error("randomized_action_scale must be positive and non_edges must be nonnegative")
    return args


def theory_sweeps_main(argv=None):
    args = diagnostics_parse_args(argv)
    out_path = Path(args.out_path) if args.out_path else Path(args.out_dir) / "theory_diagnostics.json"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(str(out_path))
    records = []
    sweep_parameters = {
        "confounding": ("outcome_confounding", args.confounding_values),
        "proxy": ("observed_proxy_strength", args.proxy_values),
        "depth": ("chain_depth", args.depth_values),
    }
    for seed in args.seeds:
        for sweep in args.sweeps:
            if sweep in sweep_parameters:
                axis, values = sweep_parameters[sweep]
                records.extend(diagnostics_chain_record(args, seed, sweep, axis, value) for value in values)
            elif sweep == "coverage":
                records.extend(diagnostics_coverage_records(args, seed))
            elif sweep == "spectral":
                records.extend(diagnostics_spectral_records(args, seed))
    aggregates, correlations = diagnostics_summarize(records)
    payload = {
        "protocol": "linear_gaussian_causal_theory_diagnostics_v1",
        "config": vars(args),
        "semantics": {
            "data": "Independent observational, randomized-training, and fixed-do evaluation RNG streams; paired random numbers across sweep conditions.",
            "scm": "U,e_A,e_P iid N(0,1); P=c*U+e_P; observational A=lambda*U+e_A; Y_0=beta*A+gamma*U+epsilon_0; Y_j=alpha*Y_(j-1)+epsilon_j.",
            "oracle": "E[Y | P, do(A=a)], integrating over latent U and all outcome innovations. No realized outcome noise or latent draw is supplied to a predictor.",
            "proxy": "c controls an additional observed pre-action proxy P; lambda separately controls observational action confounding.",
            "coverage": "Independent exposure and detection Bernoulli trials with explicit optional false matches; the union bounds concern missing true edges only.",
            "spectral": "Finite repeated composition for real symmetric propagation matrices; this recurrence is distinct from the acyclic chain SCM.",
            "reporting": "Per-seed measured and analytic diagnostics, sample standard deviations across seeds, and Spearman correlations with average ranks for ties; undefined values are null.",
            "scope": "Controlled diagnostic implementation; no reported paper metric is embedded as a target or asserted as reproduced.",
        },
        "per_seed": records,
        "aggregate": aggregates,
        "correlations": correlations,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized)
    print(str(out_path))


if __name__ == "__main__":
    if "--theory-sweeps" in sys.argv[1:]:
        theory_sweeps_main()
    else:
        main()
