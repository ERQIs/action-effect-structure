"""Run both minimal action-effect experiments from one entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = (
    ROOT / "experiments" / "one_dof",
    ROOT / "experiments" / "two_dof",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the reduced smoke-test configuration and write results_quick/.",
    )
    args = parser.parse_args()

    output_dir = "results_quick" if args.quick else "results"
    for experiment_dir in EXPERIMENTS:
        command = [
            sys.executable,
            "experiment.py",
            "--output-dir",
            output_dir,
        ]
        if args.quick:
            command.append("--quick")
        print(f"\nRunning {experiment_dir.name} experiment...")
        subprocess.run(command, cwd=experiment_dir, check=True)


if __name__ == "__main__":
    main()


