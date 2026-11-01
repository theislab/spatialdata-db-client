"""Check every cohort's marker set against each member's gene panel (Task 2, step 2b).

Run before any embedding. It reads only each kept member's ``tables/<name>/var``, so it is light, but
it reads every member over the network: run it on a compute node, not a login node.

    uv run python scripts/panel_check.py cohorts.yaml --release 2026-10-28 --out panels

Writes ``<out>.tsv`` (one row per member) and ``<out>.md`` (one row per cohort and panel, naming the
compartments a panel misses). A missing compartment needs a ``fallback`` in markers.yaml, chosen from the
panel's genes before any embedding. Exits 1 while any compartment is uncovered.

The tsv also records, per member, how many symbol-keyed features also carry an Ensembl id
(``n_ensembl``), which goes into the primer manifest.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml
from anndata.io import read_elem
from primers.cohort import WRONG_ORGANISM, WrongOrganismError, cohort_members, restrict_to_organism, spec_families
from primers.markers import coverage, load_markers, panel_genes, panel_id, uncovered


def read_panel(row: pd.Series) -> dict:
    """The member's gene symbols and the number of them that also have an Ensembl id."""
    from sddb.concat import _harmonize_var, _table_name
    from sddb.remote import _open_group

    url = f"{str(row['zarr_url']).rstrip('/')}/tables/{_table_name(row)}"
    var = ad.AnnData(var=read_elem(_open_group(url)["var"]))
    by_symbol = restrict_to_organism(_harmonize_var(var, "symbol", row["uid"])[0], str(row["organism"]))
    try:
        with_ensembl = set(_harmonize_var(var, "ensembl", row["uid"])[0].var["original_var_name"])
    except ValueError:  # no Ensembl ids at all (Allen MERSCOPE style)
        with_ensembl = set()
    genes = set(panel_genes(by_symbol.var_names))
    n_ensembl = int(
        (by_symbol.var_names.isin(list(genes)) & by_symbol.var["original_var_name"].isin(with_ensembl)).sum()
    )
    return {"symbols": sorted(genes), "n_ensembl": n_ensembl}


def check(cohorts: list[dict], markers: dict[str, dict], catalog: object, max_workers: int = 16):
    """Per-member rows and per-(cohort, panel) rows; members whose var cannot be read are reported, not raised."""
    members, panels = [], []
    for spec in cohorts:
        if spec["marker_set"] not in markers:
            print(f"skipping {spec['slug']}: no marker set {spec['marker_set']!r} in markers.yaml", file=sys.stderr)
            continue
        kept = cohort_members(catalog, spec, spec_families(spec))
        kept = kept[kept["status"] == "kept"]
        marker_set = markers[spec["marker_set"]]

        def read(row: pd.Series) -> dict:
            try:
                return read_panel(row)
            except WrongOrganismError as err:
                return {"dropped": f"{WRONG_ORGANISM}: {err}"}
            except Exception as err:  # one bad store must not stop the check
                return {"error": f"{type(err).__name__}: {err}"}

        rows = [r for _, r in kept.iterrows()]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(read, rows))
        by_panel: dict[str, dict] = {}
        for row, res in zip(rows, results, strict=True):
            out = {"cohort": spec["slug"], "uid": row["uid"], "technology": row["technology"], "family": row["family"]}
            if "error" in res or "dropped" in res:
                members.append({**out, **res})
                continue
            pid = panel_id(row["technology"], res["symbols"])
            n = len(res["symbols"])
            members.append(
                {
                    **out,
                    "panel_id": pid,
                    "n_genes": n,
                    "n_ensembl": res["n_ensembl"],
                    "ensembl_coverage": round(res["n_ensembl"] / n, 4) if n else 0.0,
                }
            )
            by_panel.setdefault(pid, {"symbols": res["symbols"], "technology": row["technology"], "n": 0})["n"] += 1
        for pid, p in by_panel.items():
            hits = coverage(marker_set, p["symbols"])
            panels.append(
                {
                    "cohort": spec["slug"],
                    "marker_set": spec["marker_set"],
                    "panel_id": pid,
                    "members": p["n"],
                    "covered": sum(bool(h) for h in hits.values()),
                    "compartments": len(hits),
                    "uncovered": ", ".join(uncovered(marker_set, p["symbols"], pid)),
                    "markers_found": "; ".join(f"{k}: {' '.join(v)}" for k, v in hits.items() if v),
                }
            )
    return pd.DataFrame(members), pd.DataFrame(panels)


def markdown_table(df: pd.DataFrame) -> str:
    """A GitHub markdown table, without the tabulate dependency that ``DataFrame.to_markdown`` needs."""
    lines = [" | ".join(df.columns), " | ".join("---" for _ in df.columns)]
    lines += [" | ".join(str(v) for v in row) for row in df.itertuples(index=False)]
    return "\n".join(f"| {line} |" for line in lines) + "\n"


def main() -> int:
    """Run the check from the command line; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cohorts", type=Path)
    ap.add_argument("--markers", type=Path, default=None)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--release")
    src.add_argument("--url", help="a local catalog parquet")
    ap.add_argument("--out", type=Path, required=True, help="output stem; writes <out>.tsv and <out>.md")
    ap.add_argument("--include-reserve", action="store_true")
    args = ap.parse_args()

    import sddb

    catalog = sddb.Catalog.from_file(args.url) if args.url else sddb.Catalog(version=args.release)
    cohorts = [
        c for c in yaml.safe_load(args.cohorts.read_text())["cohorts"] if args.include_reserve or not c.get("reserve")
    ]
    markers = load_markers(args.markers) if args.markers else load_markers()
    members, panels = check(cohorts, markers, catalog)
    members.to_csv(args.out.with_suffix(".tsv"), sep="\t", index=False)
    args.out.with_suffix(".md").write_text(markdown_table(panels))
    n_err = int(members["error"].notna().sum()) if "error" in members else 0
    gaps = panels[panels["uncovered"] != ""] if len(panels) else panels
    print(
        f"{len(members)} members, {len(panels)} cohort panels, {len(gaps)} with uncovered compartments, {n_err} unreadable"
    )
    return 1 if len(gaps) or n_err or not len(panels) else 0


if __name__ == "__main__":
    sys.exit(main())
