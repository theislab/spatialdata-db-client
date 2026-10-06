"""Pure, JSON-serializable tool functions behind the MCP server (no ``mcp`` import)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from sddb import _catalog_schema as schema
from sddb.catalog import Catalog
from sddb.dataset import Dataset, Results

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
