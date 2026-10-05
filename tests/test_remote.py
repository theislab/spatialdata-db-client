from __future__ import annotations

import pytest
from tests._fixtures import make_tiny_sdata_zarr

from sddb._cache import cache_dir
from sddb.remote import elements, open_sdata, storage_options

PUBLIC_URL = "s3://scverse-spatial-eu-central-1/.lamindb/0EQmsj25jtIQuFUT.zarr"


def test_storage_options():
    assert storage_options("s3://b/k") == {"anon": True}
    assert storage_options("https://x") == {}
    assert storage_options("/local/p") == {}


def test_elements(tmp_path):
    out = elements(str(make_tiny_sdata_zarr(tmp_path)))
    assert out["images/img"]["shape"] == (3, 8, 8)
    assert out["images/img"]["dtype"] == "uint8"
    assert out["tables/table"]["shape"] == (4, 3)
    assert out["tables/table"]["dtype"] == "float32"


def test_open_lazy(tmp_path):
    sdata = open_sdata(str(make_tiny_sdata_zarr(tmp_path)))
    assert "img" in sdata.images
    assert "table" in sdata.tables


def test_open_eager_copies(tmp_path):
    src = make_tiny_sdata_zarr(tmp_path)
    cache = tmp_path / "cache"
    sdata = open_sdata(str(src), lazy=False, cache_dir=cache)
    assert "img" in sdata.images
    assert (cache / "tiny.zarr").is_dir()


def test_cache_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "env"))
    assert cache_dir() == tmp_path / "env"
    assert (tmp_path / "env").is_dir()
    assert cache_dir(tmp_path / "arg") == tmp_path / "arg"


@pytest.mark.parametrize("fn", [elements, open_sdata])
def test_bogus_url_names_url(fn, tmp_path):
    bogus = str(tmp_path / "nope.zarr")
    with pytest.raises(FileNotFoundError, match="nope.zarr"):
        fn(bogus)


@pytest.mark.network
def test_elements_real_public_dataset_anonymously():
    els = elements(PUBLIC_URL)
    assert els
    assert any(k.startswith("tables/") for k in els)


@pytest.mark.network
@pytest.mark.xfail(
    strict=True,
    reason="spatialdata 0.8.0 read_zarr cannot open remote stores: _resolve_zarr_store isinstance(StoreLike) "
    "TypeError, Group branch FsspecStore(fs=) TypeError, and element readers use Path(store)",
)
def test_open_lazy_real_public_dataset_anonymously():
    sdata = open_sdata(PUBLIC_URL, lazy=True)
    assert len(sdata.tables) > 0
