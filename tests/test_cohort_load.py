import json

import pandas as pd
import pytest
from tests._fixtures import make_tiny_sdata_zarr

from sddb import cohort as cohort_mod
from sddb import manifest
from sddb.catalog import Catalog
from sddb.cohort import SpatialDataCohort


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cohort_mod, "_is_remote", lambda url: True)  # treat the local fixture store as remote
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


def test_dataset_to_sdata_shares_the_cohort_cache(cohort, tmp_path):
    cohort[0].to_sdata()
    assert [p.name for p in (tmp_path / "cache" / "stores").iterdir()] == ["u10000.zarr"]


def test_catalog_dataset(tmp_path):
    p = tmp_path / "c.parquet"
    pd.DataFrame(
        {
            "uid": ["u1"],
            "technology": ["Visium"],
            "assay": ["a"],
            "organism": ["human"],
            "validation_status": ["pass"],
            "zarr_url": ["s3://b/1.zarr"],
        }
    ).to_parquet(p)
    cat = Catalog.from_file(p)
    assert cat.dataset("u1").zarr_url == "s3://b/1.zarr"
    with pytest.raises(KeyError, match="u9"):
        cat.dataset("u9")


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
