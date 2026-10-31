import inspect
import re
from pathlib import Path

import pytest

from sddb import Catalog

_ROOT = Path(__file__).parent.parent
# The release the install instructions pin. To bump: replace the old tag in README.md and docs/, then change this line.
ALPHA_TAG = "v0.1.3"


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
    assert f'pip install "spatialdata-db @ git+https://github.com/theislab/spatialdata-db-client@{ALPHA_TAG}"' in index


def test_install_pins_use_one_tag():
    files = [_ROOT / "README.md", *(_ROOT / "docs").rglob("*.md"), *(_ROOT / "docs").rglob("*.ipynb")]
    files = [f for f in files if "_build" not in f.parts and "superpowers" not in f.parts]
    tags = {
        (f.relative_to(_ROOT).as_posix(), tag)
        for f in files
        for tag in re.findall(r"spatialdata-db-client@(v[\d.]+\d)|alpha release, (v[\d.]+\d)", f.read_text())
    }
    found = {(path, a or b) for path, (a, b) in tags}
    assert found, "no pinned install line found"
    assert {t for _, t in found} == {ALPHA_TAG}, sorted(p for p in found if p[1] != ALPHA_TAG)


def test_citation_cff_is_valid():
    yaml = pytest.importorskip("yaml")
    cff = yaml.safe_load((_ROOT / "CITATION.cff").read_text())
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    assert cff["title"] == "spatialdata-db"
    assert cff["authors"][0]["name"] == "Theis Lab"
    assert cff["license"] == "BSD-3-Clause"
