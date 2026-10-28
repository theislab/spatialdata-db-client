import json
import sys

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
    df = pd.DataFrame(
        {"uid": ["u10000", "u20000"], "zarr_url": [z, z], "sample_id": ["a", "a"], "size_bytes": [3_000_000, 5_000_000]}
    )
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


@pytest.mark.parametrize("with_tqdm", [True, False])
def test_download_progress_on_stderr_only_when_fetching(cohort, capsys, monkeypatch, with_tqdm):
    if not with_tqdm:
        monkeypatch.setitem(sys.modules, "tqdm", None)  # makes ``import tqdm`` fail
    cohort.to_list()
    cap = capsys.readouterr()
    assert "downloading" in cap.err
    assert cap.out == ""
    if not with_tqdm:
        assert "2 datasets, 8 MB" in cap.err
    cohort.to_list()
    assert capsys.readouterr().err == ""  # everything cached: silent


def test_download_progress_first_run_only(cohort, tmp_path, capsys):
    cohort.download(tmp_path / "d")
    assert "downloading" in capsys.readouterr().err
    cohort.download(tmp_path / "d")
    assert capsys.readouterr().err == ""  # all complete in dest: silent


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


def test_elements_cache_is_per_version(cohort, tmp_path):
    cohort.to_list(elements=["tables/table"])
    roots = sorted(p.name for p in (tmp_path / "cache" / "elements").iterdir())
    assert roots == ["u10000", "u20000"]  # same zarr_url, two uids: two partial copies, never shared


def test_elements_dest_copies_partial_and_records_it(cohort, tmp_path, capsys):
    d = tmp_path / "data"
    out = cohort.to_list(elements=["images/img"], dest=d)
    assert list(out[0].images) == ["img"]
    assert not out[0].tables
    assert "downloading" in capsys.readouterr().err
    entries = manifest.Manifest.read(d / "manifest.json").entries
    assert [(e.uid, e.status, e.elements) for e in entries] == [
        ("u10000", "partial", ["images/img"]),
        ("u20000", "partial", ["images/img"]),
    ]
    assert not (d / "u10000.zarr" / "tables").exists()
    assert not (tmp_path / "cache" / "elements").exists()
    assert manifest.reusable(d) == {}  # a partial copy is never a complete store


def test_elements_dest_reuses_and_extends_the_copy(cohort, tmp_path, capsys):
    d = tmp_path / "data"
    cohort[0].to_sdata(elements=["images/img"], dest=d)
    ino = (d / "u10000.zarr").stat().st_ino
    capsys.readouterr()
    cohort[0].to_sdata(elements=["images/img"], dest=d)
    assert capsys.readouterr().err == ""  # already there: silent, no copy
    assert (d / "u10000.zarr").stat().st_ino == ino
    sdata = cohort[0].to_sdata(elements=["tables/table"], dest=d)
    assert "table" in sdata.tables
    assert "img" in sdata.images  # the recorded elements are kept
    (e,) = manifest.Manifest.read(d / "manifest.json").entries
    assert e.elements == ["images/img", "tables/table"]


def test_full_download_replaces_a_partial_copy(cohort, tmp_path):
    d = tmp_path / "data"
    cohort.to_list(elements=["images/img"], dest=d)
    entries = cohort.download(d).entries
    assert [(e.status, e.elements) for e in entries] == [("complete", None), ("complete", None)]
    assert (d / "u10000.zarr" / "tables" / "table").is_dir()
    sdata = cohort[0].to_sdata(elements=["images/img"], dest=d)  # a complete store opens whole
    assert "table" in sdata.tables
    assert manifest.Manifest.read(d / "manifest.json").entries[0].status == "complete"


def test_refetch_copies_only_the_partial_elements(cohort, tmp_path):
    cohort[0].to_sdata(elements=["images/img"], dest=tmp_path / "data")
    m = manifest.Manifest.read(tmp_path / "data" / "manifest.json").refetch(tmp_path / "again")
    assert [(e.uid, e.status, e.elements) for e in m.entries] == [("u10000", "partial", ["images/img"])]
    assert not (tmp_path / "again" / "u10000.zarr" / "tables").exists()


def test_elements_dest_bad_path_raises_and_keeps_the_copy(cohort, tmp_path):
    d = tmp_path / "data"
    cohort[0].to_sdata(elements=["images/img"], dest=d)
    with pytest.raises(FileNotFoundError, match="images/nope"):
        cohort[0].to_sdata(elements=["images/nope"], dest=d)
    with pytest.raises(ValueError, match="kind/name"):
        cohort[0].to_sdata(elements=["img"], dest=d)
    assert (d / "u10000.zarr" / "images" / "img").is_dir()
    assert manifest.Manifest.read(d / "manifest.json").entries[0].elements == ["images/img"]


def test_manifest_without_elements_field_loads(tmp_path):
    entry = {
        "uid": "u1",
        "zarr_url": "s3://b/1.zarr",
        "dest": "x",
        "size_bytes": 1,
        "status": "complete",
        "sha256": "h",
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"generated_at": "t", "catalog_version": None, "entries": [entry]}))
    (e,) = manifest.Manifest.read(p).entries
    assert (e.status, e.elements) == ("complete", None)
