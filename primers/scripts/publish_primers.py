"""Stage primer bundles as GitHub release assets and print the release commands (Task 10). Dry run only.

    python scripts/publish_primers.py <bundles dir> --release 2026-10-28 --out <staging dir> [--slugs a,b]

Each bundle in ``<bundles dir>/<slug>/`` is checked first:

- the files the site needs (manifest.json, umap.parquet, qc.parquet, cohort.parquet) are present;
- its manifest names this catalog release and the GPU path (published numbers come from the GPU run);
- it is not a smoke run (slug ending in ``-smoke``) and its notebook carries no development banner;
- every file is under GitHub's 2 GiB release-asset limit.

Then each file is staged as one flat asset ``<slug>__<file>`` (hard-linked when possible), and
``index.json`` is written in the shape the website reads (web repo, scripts/primer-bundles.mjs):
``{"tag": "primers-<release>", "primers": [{"slug": ..., "files": {file: size in bytes}}]}``.

The script never creates a release or uploads anything. It prints the ``gh`` commands, which Tim runs.
Exits 1, naming every problem, when any bundle fails a check.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

REPO = "theislab/spatialdata-db-client"
REQUIRED = ("manifest.json", "umap.parquet", "qc.parquet", "cohort.parquet")
OPTIONAL = ("primer.ipynb", "primer.html")
MAX_ASSET_BYTES = 2 * 1024**3  # GitHub's release-asset limit


def asset_name(slug: str, file: str) -> str:
    """The flat release-asset name of one bundle file (the website's primerAssetName)."""
    return f"{slug}__{file}"


def _manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text()) if (bundle / "manifest.json").is_file() else {}


def embedding_files(bundle: Path) -> list[str]:
    """The bundle's embedding files as its manifest lists them (one embedding.h5ad, or its parts)."""
    return [p["file"] for p in _manifest(bundle).get("embedding") or [{"file": "embedding.h5ad"}]]


def check_bundle(bundle: Path, release: str) -> list[str]:
    """Problems that keep this bundle from being published (empty when it is ready)."""
    slug, problems = bundle.name, []
    if slug.endswith("-smoke"):
        problems.append("is a smoke run")
    listed = embedding_files(bundle) if "embedding" in _manifest(bundle) else []
    missing = [f for f in (*REQUIRED, *listed) if not (bundle / f).is_file()]
    if missing:
        problems.append(f"misses {', '.join(missing)}")
    if (bundle / "manifest.json").is_file():
        m = json.loads((bundle / "manifest.json").read_text())
        if m.get("release") != release:
            problems.append(f"was computed on release {m.get('release')!r}, not {release!r}")
        if m.get("path") != "gpu":
            problems.append(f"ran on the {m.get('path')!r} path; published numbers come from the GPU run")
        if m.get("slug") != slug:
            problems.append(f"manifest slug {m.get('slug')!r} differs from the folder name")
    if (bundle / "primer.ipynb").is_file():
        params = (
            json.loads((bundle / "primer.ipynb").read_text())
            .get("metadata", {})
            .get("papermill", {})
            .get("parameters", {})
        )
        if params.get("dev_note"):
            problems.append(f"is a development run: {params['dev_note']}")
    for f in (*REQUIRED, *embedding_files(bundle), *OPTIONAL):
        if (bundle / f).is_file() and (size := (bundle / f).stat().st_size) >= MAX_ASSET_BYTES:
            problems.append(f"{f} is {size} bytes, at or over the 2 GiB asset limit")
    return problems


def stage(bundles: list[Path], release: str, out: Path) -> dict:
    """Stage every file of ``bundles`` under its asset name in ``out`` and write ``out/index.json``."""
    out.mkdir(parents=True, exist_ok=True)
    index = {"tag": f"primers-{release}", "primers": []}
    for bundle in sorted(bundles, key=lambda b: b.name):
        files = {}
        for f in (*REQUIRED, *embedding_files(bundle), *OPTIONAL):
            src = bundle / f
            if not src.is_file():
                continue
            dest = out / asset_name(bundle.name, f)
            dest.unlink(missing_ok=True)
            try:
                os.link(src, dest)
            except OSError:  # another file system: copy instead
                shutil.copy2(src, dest)
            files[f] = src.stat().st_size
        index["primers"].append({"slug": bundle.name, "files": files})
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def commands(index: dict, out: Path) -> list[str]:
    """The gh commands that create the release and upload the staged assets (never run here)."""
    tag = index["tag"]
    assets = [str(out / "index.json")] + [
        str(out / asset_name(p["slug"], f)) for p in index["primers"] for f in p["files"]
    ]
    create = f"gh release create {tag} --repo {REPO} --title {shlex.quote(f'Cohort primers on catalog {tag[8:]}')} --generate-notes"
    return [create] + [f"gh release upload {tag} --repo {REPO} {shlex.quote(a)}" for a in assets]


def main() -> int:
    """Check, stage and print; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", type=Path)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slugs", help="comma-separated slugs; default every bundle folder")
    args = ap.parse_args()
    wanted = set(args.slugs.split(",")) if args.slugs else None
    bundles = [b for b in sorted(args.bundles.iterdir()) if b.is_dir() and (wanted is None or b.name in wanted)]
    if wanted and (absent := wanted - {b.name for b in bundles}):
        print(f"no bundle folder for {', '.join(sorted(absent))}", file=sys.stderr)
        return 1
    problems = {b.name: p for b in bundles if (p := check_bundle(b, args.release))}
    if problems:
        for slug, ps in problems.items():
            print(f"{slug}: " + "; ".join(ps), file=sys.stderr)
        return 1
    index = stage(bundles, args.release, args.out)
    print(f"staged {len(index['primers'])} primers in {args.out}; nothing was released or uploaded.")
    print("To publish (Tim):")
    for c in commands(index, args.out):
        print("  " + c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
