import re
from pathlib import Path

import sddb

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
