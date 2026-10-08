"""MCP stdio server: thin shell over :mod:`sddb.mcp.tools`."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from sddb.mcp import tools


def _server_class() -> Any:
    """Return the MCP server class (``MCPServer`` in mcp>=2, ``FastMCP`` in 1.x)."""
    for mod, name in (("mcp.server.mcpserver", "MCPServer"), ("mcp.server.fastmcp", "FastMCP")):
        try:
            return getattr(importlib.import_module(mod), name)
        except (ImportError, AttributeError):
            continue
    raise ImportError("the MCP server needs the optional extra: pip install 'spatialdata-db[mcp]'")


def _drop_none(**kw: Any) -> dict[str, Any]:
    """Keyword args with ``None`` values removed — the wrappers forward only what the caller set."""
    return {k: v for k, v in kw.items() if v is not None}


def build_server() -> Any:
    """Create the FastMCP server with the catalog tools registered."""
    server = _server_class()("spatialdata-db")

    def query(
        organism: str | None = None,
        tissue: str | None = None,
        disease: str | None = None,
        technology: str | None = None,
        assay: str | None = None,
        tier: str | None = None,
        validation: str = "pass",
        expressing: str | None = None,
        min_fraction: float | None = None,
        search: str | None = None,
        min_obs: int | None = None,
        min_features: int | None = None,
    ) -> list[dict[str, Any]]:
        return tools.query_tool(
            validation=validation,
            **_drop_none(
                organism=organism, tissue=tissue, disease=disease,
                technology=technology, assay=assay, tier=tier,
                n_obs__gte=min_obs, n_features__gte=min_features,
                expressing=expressing, min_fraction=min_fraction, search=search,
            ),
        )

    def describe(uid: str) -> dict[str, Any]:
        return tools.describe_tool(uid)

    def genes(
        symbols: list[str],
        organism: str | None = None,
        tissue: str | None = None,
        disease: str | None = None,
        assay: str | None = None,
        mode: str = "all",
        include_bronze: bool = True,
    ) -> list[dict[str, Any]]:
        return tools.genes_tool(
            symbols, mode=mode, include_bronze=include_bronze,
            **_drop_none(organism=organism, tissue=tissue, disease=disease, assay=assay),
        )

    def facets(field: str | None = None) -> list[str]:
        return tools.facets_tool(field)

    def cite(
        organism: str | None = None,
        tissue: str | None = None,
        disease: str | None = None,
        technology: str | None = None,
        assay: str | None = None,
        tier: str | None = None,
        search: str | None = None,
        expressing: str | None = None,
        min_fraction: float | None = None,
        validation: str = "pass",
    ) -> dict[str, Any]:
        return tools.cite_tool(
            validation=validation,
            **_drop_none(
                organism=organism, tissue=tissue, disease=disease,
                technology=technology, assay=assay, tier=tier,
                search=search, expressing=expressing, min_fraction=min_fraction,
            ),
        )

    def download_cohort(
        organism: str | None = None,
        tissue: str | None = None,
        disease: str | None = None,
        technology: str | None = None,
        assay: str | None = None,
        tier: str | None = None,
        search: str | None = None,
        expressing: str | None = None,
        min_fraction: float | None = None,
        validation: str = "pass",
        download: bool = False,
    ) -> dict[str, Any]:
        return tools.download_tool(
            validation=validation, download=download,
            **_drop_none(
                organism=organism, tissue=tissue, disease=disease,
                technology=technology, assay=assay, tier=tier,
                search=search, expressing=expressing, min_fraction=min_fraction,
            ),
        )

    server.tool(
        name="query",
        description="Filter datasets by facets, ranges (min_obs/min_features), gene (expressing) and free-text search.",
    )(query)
    server.tool(name="describe", description="Catalog row and element shapes for one dataset uid.")(describe)
    server.tool(
        name="genes",
        description=(
            "Datasets expressing one or more genes (symbols or Ensembl ids; mode all=AND, any=union), "
            "optionally filtered by organism/tissue/disease/assay. Bronze datasets are included by default."
        ),
    )(genes)
    server.tool(name="facets", description="List facet columns, or the distinct values of a facet field.")(facets)
    server.tool(name="cite", description="BibTeX for the studies of a filtered cohort.")(cite)
    server.tool(name="download", description="Plan (sizes) or fetch a filtered cohort to the server download dir.")(download_cohort)
    return server


def main() -> None:
    """Console entry point (``sddb-mcp``): serve over stdio."""
    try:
        server = build_server()
    except ImportError as err:
        sys.exit(str(err))
    server.run()
