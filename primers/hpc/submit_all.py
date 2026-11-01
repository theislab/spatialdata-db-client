"""Print, or submit, one run_primer.sbatch job per cohort, each waiting for the previous one to end.

    python hpc/submit_all.py [--submit] [--only human-colon,human-kidney] [--cohorts cohorts.yaml]

Cohorts marked ``reserve: true`` are left out. Without ``--submit`` the sbatch commands are only printed,
with ``<job id of SLUG>`` standing for the job each one waits for. The jobs read RELEASE, SMOKE and
PUBLISH from the environment (``--export=ALL``).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SBATCH = HERE / "run_primer.sbatch"


def slugs(cohorts: Path, only: list[str] | None = None) -> list[str]:
    """The slugs of the cohorts that are not reserve, in file order, limited to ``only`` when given."""
    found = [c["slug"] for c in yaml.safe_load(cohorts.read_text())["cohorts"] if not c.get("reserve")]
    if only:
        if unknown := sorted(set(only) - set(found)):
            raise SystemExit(f"not a non-reserve cohort: {', '.join(unknown)}")
        found = [s for s in found if s in only]
    return found


def command(slug: str, after: str | None = None) -> list[str]:
    """The sbatch command for one cohort, waiting for job ``after`` to end when given."""
    dependency = [f"--dependency=afterany:{after}"] if after else []
    return ["sbatch", "--parsable", *dependency, f"--export=ALL,SLUG={slug}", str(SBATCH)]


def main() -> int:
    """Print or submit the chain of jobs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohorts", type=Path, default=HERE.parent / "cohorts.yaml")
    ap.add_argument("--only", default="", help="comma-separated slugs")
    ap.add_argument("--submit", action="store_true", help="submit the jobs instead of printing the commands")
    args = ap.parse_args()
    previous = None
    for slug in slugs(args.cohorts, [s for s in args.only.split(",") if s]):
        if args.submit:
            out = subprocess.run(command(slug, previous), check=True, capture_output=True, text=True).stdout
            previous = out.strip().split(";")[0]
            print(f"{slug}: job {previous}")
        else:
            print(shlex.join(command(slug, f"<job id of {previous}>" if previous else None)))
            previous = slug
    return 0


if __name__ == "__main__":
    sys.exit(main())
