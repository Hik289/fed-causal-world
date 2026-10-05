import argparse
from pathlib import Path
import runpy
import sys


EXPERIMENTS = {
    "theory": ("causal_theory_sweeps.py", "Confounding, observed proxies, depth, interface coverage, and spectral stability."),
    "interfaces": ("interface_construction.py", "Recover interfaces from supplied intervention-response traces and measure construction costs."),
    "modular": ("modular_agent_evaluation.py", "ALFWorld and tau-bench graph, control, attention, feedback, model, and efficiency evaluations."),
    "scienceworld": ("scienceworld_evaluation.py", "Installed ScienceWorld tasks and official scores with supplied interface graphs."),
    "apibank": ("apibank_evaluation.py", "APIBank persistent-state dialogues scored with the official API-call checker."),
    "summarize": ("summarize_agent_runs.py", "Aggregate saved benchmark outputs and compare matched evaluation tasks."),
}


def main():
    parser = argparse.ArgumentParser(
        description="Experiment entrypoints for the current manuscript. Each command requires its own data and configuration; no published results are embedded.",
        epilog="Pass an experiment name followed by --help for its arguments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiment", choices=EXPERIMENTS, help="\n".join(f"{key}: {value[1]}" for key, value in EXPERIMENTS.items()))
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    arguments = parser.parse_args(sys.argv[1:2])
    target = Path(__file__).resolve().parent / EXPERIMENTS[arguments.experiment][0]
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
