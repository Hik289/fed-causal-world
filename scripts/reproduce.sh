#!/usr/bin/env bash

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/synthetic runs

echo "[1/6] Generating synthetic SCM data ..."
for seed in 0 1 2; do
  python src/fed_causal/synthetic_scm_skeleton.py \
    --config_id "sanity_chain_d4_seed${seed}" \
    --K 5 --d 4 --alpha 0.8 --gamma 0.0 --sigma 0.05 \
    --T_train 3000 --T_eval 1500 --seed "$seed" --force_chain \
    --out_dir data/synthetic
  python src/fed_causal/synthetic_scm_skeleton.py \
    --config_id "medium_chain_d4_seed${seed}" \
    --K 5 --d 4 --alpha 0.5 --gamma 0.5 --sigma 0.10 \
    --T_train 3000 --T_eval 1500 --seed "$seed" --force_chain \
    --out_dir data/synthetic
  python src/fed_causal/synthetic_scm_skeleton.py \
    --config_id "confact_chain_d4_seed${seed}" \
    --K 5 --d 4 --alpha 0.5 --gamma 0.5 --sigma 0.10 \
    --T_train 3000 --T_eval 1500 --seed "$seed" --force_chain \
    --confound_action --gamma_A 2.0 \
    --out_dir data/synthetic
done

echo "[2/6] Theorem 1 verification (γ-sweep, confounder shift) ..."
python experiments/anchor4_v9_b10fix.py \
  --data_dir data/synthetic \
  --out_dir runs/anchor_4_v9

echo "[3/6] Theorem 2 verification (ICP edge validation, N_min) ..."
python experiments/anchor4_v5_icp.py \
  --data_dir data/synthetic \
  --out_dir runs/anchor_4_v5

echo "[4/6] Proposition 1 verification (horizon B10-B2 gap) ..."
python experiments/anchor4_v10_horizon.py \
  --data_dir data/synthetic \
  --out_dir runs/anchor_4_v10

echo "[5/6] P1.5: 3-seed G6 Spearman correlations ..."
python experiments/p15_synthetic_3seed.py \
  --out_dir runs/p15_synthetic_3seed

echo "[6/6] P2.2: do-calculus B8' (Pearl §3.3 single-proxy back-door) ..."
python experiments/p22_docalculus_synthetic.py \
  --out_dir runs/p22_docalculus

echo
echo "Done. Summaries in:"
echo "  runs/anchor_4_v9/"
echo "  runs/anchor_4_v5/"
echo "  runs/anchor_4_v10/"
echo "  runs/p15_synthetic_3seed/"
echo "  runs/p22_docalculus/"
echo
echo "For the agentic experiments (τ-bench, ALFWorld), see README §4."
