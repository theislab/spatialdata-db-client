"""Pure, JSON-serializable tool functions behind the MCP server (no ``mcp`` import)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import platformdirs

from sddb.catalog import Catalog, facet_values
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset
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
    ``**facets`` accepts equality, lists (isin) and ``col__gte``/``gt``/``lte``/``lt`` ranges. ``expressing``
    (+ optional ``min_fraction``) and ``search`` compose as in :meth:`Catalog.query`.
    """
    df = _query(
        catalog_url, validation=validation, expressing=expressing,
        min_fraction=min_fraction, search=search, **facets,
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
    facets = {k: v for k, v in (("organism", organism), ("tissue", tissue), ("disease", disease), ("assay", assay)) if v is not None}
    cohort = Catalog(catalog_url).query(
        expressing=symbols, mode=mode, validation=None if include_bronze else "pass", **facets
    )
    return _records(cohort.to_df())


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
    """Return BibTeX text + entry count for the studies of a filtered cohort (empty cohort -> n=0)."""
    res = _query(
        catalog_url, validation=validation, expressing=expressing,
        min_fraction=min_fraction, search=search, **facets,
    )
    if len(res) == 0:
        return {"bibtex": "", "n": 0}
    text, n = res.citations_text(bib_url=bib_url)
    return {"bibtex": text, "n": n}


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
        catalog_url, validation=validation, expressing=expressing,
        min_fraction=min_fraction, search=search, **facets,
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
