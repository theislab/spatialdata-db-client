import inspect
from pathlib import Path

import pytest

from sddb import Catalog

_ROOT = Path(__file__).parent.parent


def test_quickstart_api_exists():
    cat = Catalog.from_file(_ROOT / "docs" / "_data" / "catalog_sample.parquet")
    hits = cat.query(technology="Visium")
    assert len(hits) >= 1
    assert hasattr(hits, "summary")
    assert min(hits, key=lambda ds: ds.size_bytes) is not None
    assert "lazy" in inspect.signature(type(hits[0]).load).parameters
    index = (_ROOT / "docs" / "index.md").read_text()
    assert "load(lazy=False)" in index
    assert "min(hits, key=lambda ds: ds.size_bytes)" in index
    assert "pip install git+https://github.com/theislab/spatialdata-db-client" in index


def test_citation_cff_is_valid():
    yaml = pytest.importorskip("yaml")
    cff = yaml.safe_load((_ROOT / "CITATION.cff").read_text())
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["title"] == "spatialdata-db"
    assert cff["authors"][0]["family-names"] == "Treis"
    assert cff["license"] == "BSD-3-Clause"
