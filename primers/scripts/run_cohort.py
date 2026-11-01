"""Build one primer bundle: load the cohort, QC each family, embed the views and export (Tasks 4 to 6).

    python scripts/run_cohort.py cohorts.yaml human-colon --release 2026-10-28 --out bundles [--cpu] [--smoke 3]

``--smoke N`` keeps only the smallest kept member of each of N papers, taken in turn from each family, for a
quick end-to-end check; the bundle is written as ``<slug>-smoke``. Phase timings and the peak host memory
are printed and added to the manifest as ``phases_s`` and ``peak_rss_gb``.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from primers.cohort import load_cohort, smoke_spec
from primers.embed import embed_views
from primers.export import export_bundle
from primers.markers import load_markers
from primers.qc import apply_qc, load_thresholds, qc_floors


def main() -> int:
    """Run the cohort named on the command line and print where the bundle went."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cohorts", type=Path)
    ap.add_argument("slug")
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cpu", action="store_true", help="the CPU path (tests and checks only; published numbers are GPU)"
    )
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    import sddb

    start, phases = time.time(), {}

    def phase(name: str) -> None:
        phases[name] = round(time.time() - start - sum(phases.values()), 1)
        print(f"{name}: {phases[name]} s", flush=True)

    if not args.cpu:
        import cupy as cp
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator

        rmm.reinitialize(managed_memory=True)
        cp.cuda.set_allocator(rmm_cupy_allocator)
    spec = next(c for c in yaml.safe_load(args.cohorts.read_text())["cohorts"] if c["slug"] == args.slug)
    catalog = sddb.Catalog(version=args.release)
    if args.smoke:
        spec = smoke_spec(spec, catalog, args.smoke)
    adatas, cohort = load_cohort(spec, catalog=catalog)
    phase("load")
    thresholds = load_thresholds()
    qcs = []
    for fam in list(adatas):
        adatas[fam], q = apply_qc(adatas[fam], thresholds)
        qcs.append(q)
    qc = pd.concat(qcs, ignore_index=True)
    phase("qc")
    views = embed_views(adatas, load_markers()[spec["marker_set"]], floor=qc_floors(thresholds), use_gpu=not args.cpu)
    phase("embed")
    out = export_bundle(views, cohort, qc, spec, args.out, args.release, runtime_s=time.time() - start)
    phase("export")
    peak_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)  # KiB on Linux
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(phases_s=phases, peak_rss_gb=peak_gb, runtime_s=round(time.time() - start, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"bundle {out}; views {sorted(views)}; peak host memory {peak_gb} GB; {manifest['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
