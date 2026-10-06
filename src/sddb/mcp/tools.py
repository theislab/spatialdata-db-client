"""Pure, JSON-serializable tool functions behind the MCP server (no ``mcp`` import)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import platformdirs

from sddb import _catalog_schema as schema
from sddb.catalog import Catalog
from sddb.citations import parse_bibtex
from sddb.dataset import Dataset, Results
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
) -> Results:
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


def genes_tool(symbol: str, catalog_url: str | None = None) -> list[dict[str, Any]]:
    """Return catalog rows of datasets whose gene index contains ``symbol``."""
    return _records(Catalog(catalog_url).genes.datasets_with(symbol).to_df())


def facets_tool(field: str | None = None, catalog_url: str | None = None) -> list[str]:
    """List facet columns present in the catalog, or the sorted distinct values of ``field``."""
    df = Catalog(catalog_url).to_df()
    if field is None:
        return [c for c in schema.FACETS if c in df.columns]
    if field not in schema.FACETS or field not in df.columns:
        raise ValueError(f"unknown facet {field!r}; valid facets: {list(schema.FACETS)}")
    return sorted(str(v) for v in df[field].dropna().unique())


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
    with tempfile.TemporaryDirectory() as d:
        path = res.citations(Path(d) / "c.bib", bib_url=bib_url)
        text = path.read_text(encoding="utf-8")
    return {"bibtex": text, "n": len(parse_bibtex(text))}


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
