"""Pure, JSON-serializable tool functions behind the MCP server (no ``mcp`` import)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import platformdirs

from sddb.catalog import Catalog, facet_values, resolve_facet_values
from sddb.citations import citation_entries
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset
from sddb.features import _check_mode
from sddb.manifest import plan_sizes

MAX_ROWS = 200


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(df.to_json(orient="records"))
    return out


def _query(
    catalog_url: str | None,
    *,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    **facets: Any,
) -> SpatialDataCohort:
    """Shared filter path for the MCP tools: one Catalog.query call with the WP-A surface."""
    return Catalog(catalog_url).query(
        validation=None if validation == "all" else validation,
        expressing=expressing,
        min_fraction=min_fraction,
        search=search,
        **facets,
    )


def query_tool(
    catalog_url: str | None = None,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    **facets: Any,
) -> list[dict[str, Any]]:
    """Query the catalog by facets; at most ``MAX_ROWS`` rows (a trailing ``{"note": ...}`` marks truncation).

    ``validation`` filters ``validation_status`` (default ``"pass"``; ``"all"`` includes every row).
    ``**facets`` accepts equality, lists (isin) and ``col__gte``/``gt``/``lte``/``lt`` ranges, plus the
    ``noncommercial`` and ``redistributable`` keywords. ``expressing`` (+ optional ``min_fraction``) and
    ``search`` compose as in :meth:`Catalog.query`, which also resolves facet aliases.
    """
    df = _query(
        catalog_url,
        validation=validation,
        expressing=expressing,
        min_fraction=min_fraction,
        search=search,
        **facets,
    ).to_df()
    rows = _records(df.head(MAX_ROWS))
    if len(df) > MAX_ROWS:
        rows.append({"note": f"truncated: showing {MAX_ROWS} of {len(df)} rows; narrow the query"})
    return rows


def describe_tool(uid: str, catalog_url: str | None = None) -> dict[str, Any]:
    """Return the catalog row for ``uid`` plus element shapes (``elements`` is None if remote open fails)."""
    cat = Catalog(catalog_url)
    df = cat.to_df()
    df = df[df["uid"] == uid]
    if df.empty:
        raise ValueError(f"uid not in catalog: {uid}")
    row = _records(df)[0]
    try:
        row["elements"] = json.loads(json.dumps(Dataset(df.iloc[0]).elements(), default=list))
    except Exception as err:
        row["elements"] = None
        row["elements_error"] = str(err)
    return row


def genes_tool(
    symbols: list[str],
    organism: str | None = None,
    tissue: str | None = None,
    disease: str | None = None,
    assay: str | None = None,
    mode: str = "all",
    include_bronze: bool = True,
    catalog_url: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog rows of datasets expressing ``symbols`` (symbols or Ensembl ids; ``mode`` all/any).

    Bronze/non-pass datasets are included by default (``include_bronze=False`` keeps only validated ones);
    ``organism``/``tissue``/``disease``/``assay`` narrow the result.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    facets: dict[str, Any] = {
        k: v
        for k, v in (("organism", organism), ("tissue", tissue), ("disease", disease), ("assay", assay))
        if v is not None
    }
    cohort = Catalog(catalog_url).query(
        expressing=symbols, mode=mode, validation=None if include_bronze else "pass", **facets
    )
    return _records(cohort.to_df())


def features_tool(
    features: list[str] | str,
    modality: str | None = None,
    mode: str = "all",
    feature_type: str | None = None,
    organism: str | None = None,
    tissue: str | None = None,
    disease: str | None = None,
    assay: str | None = None,
    include_bronze: bool = True,
    catalog_url: str | None = None,
) -> list[dict[str, Any]]:
    """Catalog rows of datasets measuring ``features`` (genes, protein targets, metabolites; ``mode`` all/any).

    ``features`` are labels or reference ids (symbol, Ensembl, UniProt, HMDB...). ``modality``/``feature_type``
    restrict the matched features. Each row carries ``features``: the typed hits
    (``raw_label``/``feature_type``/``modality``) that matched in that dataset. Bronze datasets are included by
    default; ``organism``/``tissue``/``disease``/``assay`` narrow the result.
    """
    if isinstance(features, str):
        features = [features]
    facets: dict[str, Any] = {
        k: v
        for k, v in (("organism", organism), ("tissue", tissue), ("disease", disease), ("assay", assay))
        if v is not None
    }
    _check_mode(mode)
    cat = Catalog(catalog_url)
    full = cat.to_df()
    facets = {
        col: resolve_facet_values(full, col, val if isinstance(val, list) else [val]) for col, val in facets.items()
    }
    fi = cat.features
    typed = ("raw_label", "feature_type", "modality")
    # One scan serves both the cohort and the per-dataset typed hits.
    hits = fi._uid_hits(features, columns=("uid", *typed), feature_type=feature_type, modality=modality)
    by_key = {k: g.sort_values(list(typed), kind="stable") for k, g in hits.groupby("_key")}
    per_token = [by_key.get(f.strip().casefold(), hits.iloc[:0]) for f in features]
    uids = fi._combine([set(g["uid"].astype(str)) for g in per_token], mode)
    df = fi._cohort(uids, None if include_bronze else "pass").to_df()
    for col, vals in facets.items():
        df = df[df[col].isin(vals)]
    per_uid: dict[str, list[dict[str, str]]] = {}
    for g in per_token:
        for uid, *vals in g[["uid", *typed]].itertuples(index=False):
            entry = {k: str(v) for k, v in zip(typed, vals, strict=True)}
            if entry not in per_uid.setdefault(str(uid), []):
                per_uid[str(uid)].append(entry)
    rows = _records(df)
    for row in rows:
        row["features"] = per_uid.get(row["uid"], [])
    return rows


def facets_tool(field: str | None = None, catalog_url: str | None = None) -> list[str]:
    """List facet columns present in the catalog, or the sorted distinct values of ``field``."""
    return facet_values(Catalog(catalog_url).to_df(), field)


def cite_tool(
    catalog_url: str | None = None,
    *,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    bib_url: str | None = None,
    **facets: Any,
) -> dict[str, Any]:
    """Return BibTeX text, entry count and fallback count for the studies of a filtered cohort (empty cohort -> n=0).

    ``n`` counts every entry; ``n_fallback`` counts the entries that cite the dataset page because the
    study has no published citation.
    """
    res = _query(
        catalog_url,
        validation=validation,
        expressing=expressing,
        min_fraction=min_fraction,
        search=search,
        **facets,
    )
    if len(res) == 0:
        return {"bibtex": "", "n": 0, "n_fallback": 0}
    text, n_published, n_fallback = citation_entries(res, bib_url=bib_url)
    return {"bibtex": text, "n": n_published + n_fallback, "n_fallback": n_fallback}


def _download_dir() -> Path:
    """Server-controlled download sandbox: $SDDB_MCP_DOWNLOAD_DIR or a platformdirs cache subdir."""
    env = os.environ.get("SDDB_MCP_DOWNLOAD_DIR")
    d = Path(env) if env else Path(platformdirs.user_cache_dir("sddb")) / "mcp-downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_tool(
    catalog_url: str | None = None,
    *,
    download: bool = False,
    workers: int = 4,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    **facets: Any,
) -> dict[str, Any]:
    """Plan (default) or fetch a filtered cohort.

    ``download=False`` returns per-store + total sizes and copies nothing. ``download=True`` fetches
    into the server download dir and returns the manifest path.
    """
    res = _query(
        catalog_url,
        validation=validation,
        expressing=expressing,
        min_fraction=min_fraction,
        search=search,
        **facets,
    )
    if not download:
        sizes = plan_sizes(res)
        return {
            "datasets": len(sizes),
            "total_bytes": sum(b for _, b in sizes),
            "per_store": [{"uid": u, "bytes": b} for u, b in sizes],
            "note": "pass download=true to fetch into the server download dir",
        }
    dest = _download_dir()
    manifest = res.download(dest, workers=workers)
    return {
        "manifest_path": str(dest / "manifest.json"),
        "dest": str(dest),
        "entries": [{"uid": e.uid, "size_bytes": e.size_bytes, "status": e.status} for e in manifest.entries],
    }
