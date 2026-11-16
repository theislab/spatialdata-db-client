"""The published contract fixture: download, check its sha256, validate it and read one batch with spatialbatch.

Marked ``network``: deselected by default (``addopts`` in pyproject.toml) and in CI (``pytest -m "not network"``).
Run with ``pytest -m network tests/test_loader_fixture.py``.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest

pytest.importorskip("pydantic")
spatialbatch = pytest.importorskip("spatialbatch")

from sddb import __version__  # noqa: E402
from sddb.contract import validate  # noqa: E402

pytestmark = pytest.mark.network

URL = "https://exampledata.scverse.org/spatialdata-db/fixtures/sddb-contract/0.1.0/atera-breast-crop.zarr.zip"
SHA256 = "7d0ea895d7a4a663398486a564d53ff6ce7056b1026c53b8a49516b3dfaadc5e"
ROOT = "sddb-contract-0.1.0-atera-breast-crop.zarr"  # the one top-level folder of the zip


@pytest.fixture(scope="module")
def fixture_collection(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fixture")
    archive = tmp / "fixture.zip"
    # the client's own User-Agent (sddb/_cache.py _http_fetch); the host refuses urllib's default one with a 403
    request = urllib.request.Request(URL, headers={"User-Agent": f"spatialdata-db-client/{__version__}"})
    with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SHA256
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp)
    return tmp / ROOT


def test_fixture_validates(fixture_collection):
    assert validate(fixture_collection) == []


def test_fixture_one_loader_batch(fixture_collection):
    loader = spatialbatch.Loader(images=["he"], labels=["cells"], crop_px=224, batch_size=32, shuffle=True)
    loader = loader.use_collection(spatialbatch.SpatialDatasetCollection(fixture_collection))
    batch = next(iter(loader))
    he, cells = np.asarray(batch["images"]["he"]), np.asarray(batch["labels"]["cells"])
    assert he.shape == (32, 3, 224, 224)
    assert he.dtype == np.uint8
    assert cells.shape == (32, 224, 224)
    assert batch["X"].shape[0] == 32
