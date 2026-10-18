import inspect
import re
import sys
from pathlib import Path

import typer.main

import sddb
import sddb.cli
from sddb.mcp import tools

ROOT = Path(__file__).parent.parent
REF = ROOT / "docs" / "reference"


def test_api_page_lists_all_exports():
    text = (REF / "api.md").read_text()
    for name in sddb.__all__:
        if name != "__version__":
            assert name in text, name


def test_mcp_page_lists_all_tools():
    names = re.findall(r'server\.tool\(\s*name="(\w+)"', (ROOT / "src/sddb/mcp/server.py").read_text())
    assert names
    text = (REF / "mcp.md").read_text()
    for name in names:
        assert f"`{name}`" in text, name


def test_mcp_page_lists_tool_parameters():
    text = (REF / "mcp.md").read_text()
    for fn in (tools.query_tool, tools.describe_tool, tools.genes_tool, tools.facets_tool, tools.cite_tool, tools.download_tool):
        for param in inspect.signature(fn).parameters:
            if param in {"catalog_url", "bib_url", "workers", "facets"}:  # internal / not exposed by server.py
                continue
            assert param in text, (fn.__name__, param)


def test_cli_page_lists_all_commands():
    sys.path.insert(0, str(ROOT / "docs" / "_ext"))
    from cli_docs import cli_markdown

    names = list(typer.main.get_command(sddb.cli.app).commands)
    assert names
    md = cli_markdown()
    for name in names:
        assert f"sddb-cli {name}" in md, name
