"""Run the Part 1 project checks and print a PASS/FAIL report.

Usage:
    python check.py           # all checks: subsystem sign tests + Simulations 1-4
    python check.py --fast    # subsystem sign tests only (seconds)

The same checks run via ``pytest`` and in ``notebooks/part_1_demo.ipynb``
(where every simulation is also plotted); see ``simulation/checks.py`` for
what each check verifies.  The exit code is non-zero unless everything
passes.  The CI workflow (``.github/workflows/check.yml``) runs ``pytest``,
which reports unimplemented subsystems as skipped instead of failed, so a
fresh template starts green on GitHub.
"""
import argparse
import sys

from simulation.checks import run_all, print_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="run only the fast subsystem checks")
    args = parser.parse_args()
    return 0 if print_report(run_all(fast_only=args.fast)) else 1


if __name__ == "__main__":
    sys.exit(main())
