import inspect
import json

import pandas as pd
import pytest
from tests._fixtures import make_tiny_sdata_zarr

from sddb import manifest, remote
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset


def test_dataset_load_accepts_elements(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        remote,
        "open_sdata",
        lambda url, lazy=True, elements=None: seen.update(url=url, lazy=lazy, elements=elements) or "SD",
    )
    d = Dataset(pd.Series({"uid": "u1", "zarr_url": "s3://b/1.zarr"}))
    assert d.load(lazy=False, elements=["images/he"]) == "SD"
    assert seen == {"url": "s3://b/1.zarr", "lazy": False, "elements": ["images/he"]}


def test_load_elements_param_present():
    assert "elements" in inspect.signature(Dataset.load).parameters


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    z = str(make_tiny_sdata_zarr(tmp_path))
    df = pd.DataFrame({"uid": ["u10000", "u20000"], "zarr_url": [z, z], "sample_id": ["a", "a"]})
    return SpatialDataCohort(df)


def test_to_list_caches_by_uid_and_reuses(cohort, tmp_path, monkeypatch):
    copies = []
    real = manifest._copy_store
    monkeypatch.setattr(manifest, "_copy_store", lambda url, dest: copies.append(dest) or real(url, dest))
    first = cohort.to_list()
    assert len(first) == 2
    assert "img" in first[0].images
    assert sorted(p.name for p in (tmp_path / "cache" / "stores").iterdir()) == ["u10000.zarr", "u20000.zarr"]
    cohort.to_list()
    assert len(copies) == 2  # second call opens the cached stores, no copy


def test_to_dict_keys(cohort):
    assert list(cohort.to_dict()) == ["u10000", "u20000"]
    with pytest.raises(ValueError, match="sample_id"):
        cohort.to_dict(key="sample_id")  # repeats
    with pytest.raises(KeyError, match="nope"):
        cohort.to_dict(key="nope")


def test_to_list_dest_writes_download_layout(cohort, tmp_path):
    out = cohort.to_list(dest=tmp_path / "data")
    assert len(out) == 2
    entries = json.loads((tmp_path / "data" / "manifest.json").read_text())["entries"]
    assert [e["status"] for e in entries] == ["complete", "complete"]
    with pytest.raises(ValueError, match="dest"):
        cohort.to_list(elements=["images/img"], dest=tmp_path / "data")
