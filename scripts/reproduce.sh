#!/usr/bin/env bash
# reproduce.sh — One-shot driver for FedCausalWorld theoretical experiments.
#
# Reproduces (CPU-only, no API calls, ~10 min total):
#   - Theorem 1 (anchor-4 v9 confshift)
#   - Theorem 2 (anchor-4 v5 ICP F-test)
#   - Proposition 1 (anchor-4 v10 horizon)
#   - G6 3-seed Synthetic correlations (P1.5)
#   - Do-calculus B8' Pearl §3.3 (P2.2)
#
# For the agentic experiments (τ-bench retail / airline, ALFWorld), fill
# in API credentials in src/fed_causal/llm_client.py first; see README §2.2.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/synthetic runs

# ----- 1. Generate Synthetic SCM data -----
echo "[1/6] Generating synthetic SCM data ..."
python src/fed_causal/synthetic_scm_skeleton.py \
  --config_id default \
  --n_train 1000 --n_test 500 \
  --out_dir data/synthetic

# ----- 2. Theorem 1: anchor-4 v9 confshift -----
echo "[2/6] Theorem 1 verification (γ-sweep, confounder shift) ..."
python experiments/anchor4_v9_b10fix.py \
  --data_dir data/synthetic \
  --out_dir runs/anchor_4_v9

# ----- 3. Theorem 2: anchor-4 v5 ICP F-test -----
echo "[3/6] Theorem 2 verification (ICP edge validation, N_min) ..."
python experiments/anchor4_v5_icp.py \
  --data_dir data/synthetic \
  --out_dir runs/anchor_4_v5

# ----- 4. Proposition 1: horizon scaling -----
echo "[4/6] Proposition 1 verification (horizon B10-B2 gap) ..."
python experiments/anchor4_v10_horizon.py \
  --data_dir data/synthetic \
  --out_dir runs/anchor_4_v10

# ----- 5. G6 3-seed Synthetic correlations (P1.5) -----
echo "[5/6] P1.5: 3-seed G6 Spearman correlations ..."
python experiments/p15_synthetic_3seed.py \
  --out_dir runs/p15_synthetic_3seed

# ----- 6. P2.2: do-calculus B8' Pearl §3.3 -----
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
