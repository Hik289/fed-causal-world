# Artifact Guide

This guide maps the public `fed-causal-world` repository to a reviewer-friendly artifact workflow for `FedCausalWorld`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
python experiments/anchor4_sanity.py
python experiments/anchor4_sanity_v2.py
```

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

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

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
