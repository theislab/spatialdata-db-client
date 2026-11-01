"""Summarise candidate cohorts from the catalog alone (CPU only, no data or gene index is loaded).

For each cohort in a YAML file, run its query, keep redistributable members, and report the member
count, papers (by the cohort rule in ``primers.cohort``), technologies, licences, observations and a
shared-gene estimate from ``n_features``.

Usage::

    uv run --directory primers python scripts/cohort_report.py cohorts.yaml --url <catalog.parquet> --out report.md
    uv run --directory primers python scripts/cohort_report.py cohorts.yaml --version 2026-10-31-2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from primers.cohort import MIN_PAPERS, family_query, paper_for_members, spec_families

import sddb


def cohort_members(catalog: sddb.Catalog, spec: dict) -> pd.DataFrame:
    """Rows of the cohort: query plus include_uids, minus exclude_uids, redistributable only."""
    families = spec_families(spec)
    techs = [t for fam in families.values() for t in fam]
    df = catalog.query(redistributable=True, **family_query(catalog, spec, families)).to_df()
    include = spec.get("include_uids") or []
    if include:
        extra = catalog.cohort(uids=include).to_df()
        ok = extra["license_redistributable"].fillna(False).astype(bool) & ~extra["license_unknown"].fillna(
            True
        ).astype(bool)
        df = pd.concat([df, extra[ok]]).drop_duplicates("uid")
    exclude = set(spec.get("exclude_uids") or [])
    df = df[~df["uid"].isin(exclude)]
    df = df[df["technology"].isin(techs)].reset_index(drop=True)
    # The per-release paper keeps each 10x demo apart; members without any paper key are left out,
    # as load_cohort drops them.
    df["paper_unmerged"] = paper_for_members(df, merge_10x_demos=False)["paper"]
    df = paper_for_members(df)
    return df[df["paper"].notna()].reset_index(drop=True)


def _min_features(df: pd.DataFrame) -> str:
    vals = pd.to_numeric(df["n_features"], errors="coerce")
    known = vals.dropna()
    text = f"{int(known.min()):,}" if len(known) else "n/a"
    missing = int(vals.isna().sum())
    return f"{text} ({missing} unknown)" if missing else text


def summarise(spec: dict, df: pd.DataFrame) -> dict:
    """One report row for a cohort."""
    techs = df["technology"].value_counts()
    obs = pd.to_numeric(df["n_obs"], errors="coerce").fillna(0)
    families = spec_families(spec)
    genes = [f"{fam} {_min_features(df[df['technology'].isin(techs)])}" for fam, techs in families.items()]
    genes.append(f"all {_min_features(df)}")
    n_papers = df["paper"].nunique()
    demo = df["paper_source"] == "10x_demo"
    return {
        "slug": spec["slug"],
        "members": len(df),
        "papers": int(df["paper_unmerged"].nunique()),
        "papers_merged": n_papers,
        "papers_10x": int(df.loc[demo, "paper_unmerged"].nunique()),
        "technologies": ", ".join(f"{t} {n}" for t, n in techs.items()),
        "family_papers": ", ".join(
            f"{fam} {df.loc[df['technology'].isin(techs), 'paper'].nunique()}" for fam, techs in families.items()
        ),
        "noncommercial": int(df["license_noncommercial"].fillna(False).astype(bool).sum()),
        "observations": int(obs.sum()),
        "largest": int(obs.max()) if len(obs) else 0,
        "shared_genes": "; ".join(genes),
        "flag": f"fewer than {MIN_PAPERS} papers" if n_papers < MIN_PAPERS else "",
    }


def paper_detail(df: pd.DataFrame) -> pd.DataFrame:
    """One row per paper: members, technologies, observations and how the paper id was found."""
    return (
        df.groupby("paper")
        .agg(
            members=("uid", "size"),
            technologies=("technology", lambda s: ", ".join(sorted(set(s)))),
            observations=("n_obs", "sum"),
            source=("paper_source", lambda s: ", ".join(sorted(set(s)))),
        )
        .sort_values("observations", ascending=False)
    )


def render(specs: list[dict], rows: list[dict], details: dict[str, pd.DataFrame], source: str) -> str:
    """The Markdown report: the summary table, then one paper table per cohort."""
    out = [
        "| cohort | members | papers | papers (10x demos as one) | of which 10x demos | technologies | papers per family | noncommercial | observations | largest member | shared genes (estimate) | flag |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['slug']} | {r['members']} | {r['papers']} | {r['papers_merged']} | {r['papers_10x']} | {r['technologies']} | {r['family_papers']} | {r['noncommercial']} | "
            f"{r['observations']:,} | {r['largest']:,} | {r['shared_genes']} | {r['flag']} |"
        )
    out += [
        "",
        f"Catalog: `{source}`. Members are redistributable only (`license_redistributable` True, licence known).",
        "Papers follow the cohort rule: GEO series, else first collection, else study_id, with DOI propagation within a group.",
        "Shared genes is an estimate: the smallest `n_features` per technology family and overall. "
        "The true shared set is the intersection of the panels and can be smaller. "
        "Members without `n_features` in the catalog are counted as unknown.",
        "A 10x demo paper is a 10x Genomics demo release; several of them in one cohort come from one company, "
        "not independent labs, so the papers (10x demos as one) column counts them as one paper. "
        f"That column decides the flag (fewer than {MIN_PAPERS} papers) and is the batch value in the primers.",
        "Members with no DOI, GEO series, collection or study_id have no paper and are left out.",
        "",
    ]
    for spec in specs:
        out += [f"### {spec['slug']}: {spec['title']}", "", spec.get("rationale", ""), ""]
        d = details[spec["slug"]]
        out.append("| paper | members | technologies | observations | paper from |")
        out.append("|---|---:|---|---:|---|")
        for paper, r in d.iterrows():
            out.append(
                f"| {paper} | {r['members']} | {r['technologies']} | {int(r['observations']):,} | {r['source']} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    """Run the report from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cohorts", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="catalog parquet URL or local path")
    src.add_argument("--version", help="dated catalog release, e.g. 2026-10-31-2")
    ap.add_argument("--out", type=Path, help="write the Markdown report here")
    args = ap.parse_args()

    catalog = sddb.Catalog(url=args.url) if args.url else sddb.Catalog(version=args.version)
    specs = yaml.safe_load(args.cohorts.read_text())["cohorts"]
    rows, details = [], {}
    for spec in specs:
        df = cohort_members(catalog, spec)
        rows.append(summarise(spec, df))
        details[spec["slug"]] = paper_detail(df)
    text = render(specs, rows, details, args.url or f"catalog-{args.version}")
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
