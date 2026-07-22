# Artifact Guide

Operational notes for reproducing `FedCausalWorld` from the public `fed-causal-world` repository.

## Review Path

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
python experiments/anchor4_sanity.py
python experiments/anchor4_sanity_v2.py
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `bash scripts/reproduce.sh`
- `python experiments/anchor1_b2_repro.py`
- `python experiments/anchor4_sanity.py`
- `python experiments/anchor4_sanity_v2.py`
- `python experiments/anchor4_v10_horizon.py`
- `python experiments/anchor4_v3.py`
- `python experiments/anchor4_v4.py`
- `python experiments/anchor4_v5_icp.py`
- `python experiments/anchor4_v6_unseen.py`
- `python experiments/anchor4_v7_rollout.py`
- `python experiments/anchor4_v8_confact.py`
- `python experiments/anchor4_v9_b10fix.py`
- `python experiments/exp1_alfworld.py`
- `python experiments/exp1_alfworld_exp4.py`

## Figure Assets

- `fig1_pipeline_v2.png`
- `fig2_federated_loop_v2.png`
- `fig3_causal_control_v2.png`
- `fig4_intuition_v2.png`

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
