# FedCausalWorld

Official code for **FedCausalWorld: Federated Causal World Models for Agentic LLMs**.

FedCausalWorld studies whether multiple agents can build a shared world model without sharing their local interaction data. The main idea is to represent local experience using causal transition structures, aggregate stable causal relations across clients, and use the resulting federated world model to guide planning and tool use.

The repository includes experiments on **τ-bench Retail**, **τ-bench Airline**, **ALFWorld**, and synthetic structural causal model (SCM) tasks.

## Overview

Standard federated world models may combine correlations that hold only in specific clients or environments. Under client heterogeneity and distribution shift, these correlations can produce incorrect transition predictions and poor downstream decisions.

FedCausalWorld separates local transition information into causal relations and environment-specific associations. Each client estimates a local world model from its private trajectories. The server aggregates stable relations across clients and returns a shared causal world model for agent planning.

The code supports comparisons with:

- global sequential world models;
- federated models without causal filtering;
- models without intervention or control information;
- local-only models;
- oracle causal world models.

## Repository Structure

```text
fedcausalworld/
├── src/fed_causal/       # Core models, metrics, traces, and LLM interface
├── experiments/          # Synthetic and agentic experiment scripts
├── scripts/              # Reproduction scripts
├── requirements.txt
├── LICENSE
└── README.md
```

The main components are:

```text
src/fed_causal/
├── synthetic_scm_skeleton.py   # Synthetic SCM generator
├── pipeline.py                 # World-model training and aggregation
├── metrics.py                  # Evaluation metrics
├── event_traces.py             # Agent trajectory processing
├── llm_client.py               # OpenAI-compatible API client
└── baselines/                  # Baseline implementations
```

## Installation

Create a Python 3.11 environment and install the dependencies:

```bash
conda create -n fedcausal python=3.11 -y
conda activate fedcausal
pip install -r requirements.txt
```

The agent experiments require an OpenAI-compatible chat-completions endpoint. Set the API endpoint, key, and model name in the LLM configuration used by the experiment scripts.

Example:

```python
AZURE_API_KEY = "YOUR_API_KEY"
AZURE_API_BASE = "YOUR_API_ENDPOINT"
MODEL_NAME = "YOUR_MODEL_NAME"
```

## Benchmarks

### τ-bench

Install τ-bench from its official repository or package release. The code supports both the Retail and Airline domains.

### ALFWorld

Install ALFWorld and download the required environment data following the official ALFWorld instructions.

### Synthetic SCM Tasks

Synthetic datasets are generated locally and do not require external downloads.

## Quick Start

Run the synthetic experiments with:

```bash
bash scripts/reproduce.sh
```

These experiments test causal identification, distribution shift, client heterogeneity, sample size, and planning-horizon effects.

A representative τ-bench run is:

```bash
python experiments/p11_taubench_3seed.py \
  --baselines \
    B2_GlobalSeqWM \
    B7d_AnnotatedNoFraming_NoInt \
    B8d_AnnotatedNoFraming \
    B9_AnnotatedNoFramingNoControl \
    B10_OracleCausalWM \
  --seeds 0 1 2 \
  --log_dir runs/taubench_retail
```

A representative ALFWorld run is:

```bash
python experiments/p12_alf_3seed_v2.py \
  --baselines B8d_AnnotatedNoFraming \
  --n_tasks 50 \
  --seed 0 \
  --log_dir runs/alfworld
```

Additional experiment scripts are provided under `experiments/` for cross-domain evaluation, ablation studies, multi-seed analysis, and synthetic causal tests.

## Outputs

Each experiment stores task-level predictions and aggregate results in its output directory.

Typical files include:

```text
<baseline>_predictions.jsonl   # Task-level trajectories and outcomes
summary.json                   # Aggregate success and cost statistics
```

The main evaluation measures include task success rate, confidence intervals, planning errors, tool-use errors, and performance under distribution shift.

## Reproducibility

Agentic LLM experiments may vary across repeated runs because hosted model endpoints and user simulators are not fully deterministic. We recommend reporting results over multiple seeds and retaining task-level trajectories for error analysis.

Synthetic SCM experiments are deterministic when the random seed is fixed.

## License

This project is released under the MIT License.

The repository does not redistribute τ-bench or ALFWorld data. Please obtain those resources from their official repositories.

## Citation

```bibtex
@article{fedcausalworld2026,
  title   = {FedCausalWorld: Federated Causal World Models for Agentic LLMs},
  author  = {Anonymous},
  year    = {2026},
  note    = {Under review}
}
```
