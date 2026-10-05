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
    assert out["images/img"]["type"] == "images"
    assert out["tables/table"]["type"] == "tables"
    assert out["tables/table"]["shape"] == (4, 3)
    assert out["tables/table"]["dtype"] == "float32"


def test_elements_sparse_table_dtype(tmp_path):
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    from spatialdata import SpatialData

    table = ad.AnnData(X=sp.csr_matrix(np.eye(4, 3, dtype="float32")))
    path = tmp_path / "sparse.zarr"
    SpatialData(tables={"t": table}).write(path)
    out = elements(str(path))["tables/t"]
    assert out["shape"] == (4, 3)
    assert out["dtype"] == "float32"


def test_describe_multiscale_unknown_scale_keys():
    import zarr

    from sddb.remote import _describe

    g = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    g.create_array("b", shape=(2, 4, 4), dtype="uint16")
    g.create_array("a", shape=(2, 8, 8), dtype="uint16")
    assert _describe(g, "images") == {"type": "images", "shape": (2, 8, 8), "dtype": "uint16"}


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


def test_open_remote_lazy_raises_clear_error():
    # short-circuits before any network access
    with pytest.raises(NotImplementedError, match="lazy=False"):
        open_sdata("s3://bucket/some.zarr", lazy=True)


@pytest.mark.network
@pytest.mark.xfail(
    strict=True,
    reason="spatialdata 0.8.0 read_zarr cannot open remote stores (upstream)",
)
def test_canary_spatialdata_remote_read_zarr():
    # CANARY: when upstream fixes remote read_zarr this xfail flips to a failure (strict) --
    # that is the signal to re-enable remote lazy open in sddb.remote.open_sdata.
    from spatialdata import read_zarr
    from upath import UPath

    sdata = read_zarr(UPath(PUBLIC_URL, **storage_options(PUBLIC_URL)))
    assert len(sdata.tables) > 0
