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
    ) -> list[dict[str, Any]]:
        facets = {"organism": organism, "tissue": tissue, "disease": disease}
        facets |= {"technology": technology, "assay": assay, "tier": tier}
        return tools.query_tool(**{k: v for k, v in facets.items() if v is not None})

    server.tool(name="query", description="Filter datasets by facets (organism, tissue, technology, ...).")(query)

    def describe(uid: str) -> dict[str, Any]:
        return tools.describe_tool(uid)

    def genes(symbol: str) -> list[dict[str, Any]]:
        return tools.genes_tool(symbol)

    server.tool(name="describe", description="Catalog row and element shapes for one dataset uid.")(describe)
    server.tool(name="genes", description="Datasets whose gene index contains a gene symbol.")(genes)
    return server


def main() -> None:
    """Console entry point (``sddb-mcp``): serve over stdio."""
    try:
        server = build_server()
    except ImportError as err:
        sys.exit(str(err))
    server.run()
