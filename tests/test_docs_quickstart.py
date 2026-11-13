import inspect
import re
from pathlib import Path

import pytest

from sddb import Catalog

_ROOT = Path(__file__).parent.parent
# The install line README.md and docs/index.md show; the client is on PyPI as spatialdata-db.
PYPI_INSTALL = "pip install --pre spatialdata-db"


def test_quickstart_api_exists():
    cat = Catalog.from_file(_ROOT / "docs" / "_data" / "catalog_sample.parquet")
    hits = cat.query(technology="Visium")
    assert len(hits) >= 1
    assert hasattr(hits, "summary")
    assert min(hits, key=lambda ds: ds.size_bytes) is not None
    assert "dest" in inspect.signature(type(hits[0]).to_sdata).parameters
    index = (_ROOT / "docs" / "index.md").read_text()
    assert ".to_sdata()" in index
    assert "min(hits, key=lambda ds: ds.size_bytes)" in index
    assert PYPI_INSTALL in index


def test_install_lines_use_pypi():
    files = [_ROOT / "README.md", *(_ROOT / "docs").rglob("*.md"), *(_ROOT / "docs").rglob("*.ipynb")]
    files = [f for f in files if "_build" not in f.parts and "superpowers" not in f.parts]
    stale = sorted(
        f.relative_to(_ROOT).as_posix()
        for f in files
        if re.search(r"spatialdata-db-client@v[\d.]+|alpha release, v[\d.]+|not on PyPI", f.read_text())
    )
    assert not stale, f"install lines still pin a GitHub tag: {stale}"
    assert PYPI_INSTALL in (_ROOT / "README.md").read_text()


def test_citation_cff_is_valid():
    yaml = pytest.importorskip("yaml")
    cff = yaml.safe_load((_ROOT / "CITATION.cff").read_text())
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["title"] == "spatialdata-db"
    assert cff["authors"][0]["name"] == "Theis Lab"
    assert cff["license"] == "BSD-3-Clause"
