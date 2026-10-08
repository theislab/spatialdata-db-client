import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from anndata import AnnData
from geopandas import GeoDataFrame
from shapely.geometry import Point
from spatialdata.models import Image2DModel, ShapesModel, TableModel

from sddb import remote


def _shapes(n):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (n, 2))
    return ShapesModel.parse(GeoDataFrame({"geometry": [Point(*p) for p in xy], "radius": 1.0}))


def _table(n, region):
    ad = AnnData(
        X=np.random.default_rng(1).poisson(1.0, (n, 12)).astype("float32"),
        obs=pd.DataFrame({"region": pd.Categorical([region] * n), "instance_id": np.arange(n)}),
    )
    return TableModel.parse(ad, region=region, region_key="region", instance_key="instance_id")


@pytest.fixture
def vhd_store(tmp_path) -> Path:
    rng = np.random.default_rng(2)
    imgs = {
        "s_full_image": Image2DModel.parse(rng.integers(0, 255, (3, 128, 128), dtype="uint8")),
        "s_hires_image": Image2DModel.parse(rng.integers(0, 255, (3, 32, 32), dtype="uint8")),
    }
    shapes, tables = {}, {}
    for um, n in [("002um", 400), ("008um", 100), ("016um", 25)]:
        sname = f"s_square_{um}"
        shapes[sname] = _shapes(n)
        tables[f"square_{um}"] = _table(n, sname)
    store = tmp_path / "vhd.zarr"
    sd.SpatialData(images=imgs, shapes=shapes, tables=tables).write(store)
    return store


KEEP = ["images/s_full_image", "tables/square_008um", "shapes/s_square_008um"]


def test_open_elements_subset(vhd_store, tmp_path):
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=tmp_path / "cache")
    assert set(out.images) == {"s_full_image"}
    assert set(out.tables) == {"square_008um"}
    assert set(out.shapes) == {"s_square_008um"}
    assert out.tables["square_008um"].n_obs == 100


def test_open_elements_prunes_consolidated(vhd_store, tmp_path):
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=tmp_path / "c")
    assert "s_hires_image" not in out.images
    (loc,) = (tmp_path / "c").glob("vhd__*.zarr")
    keys = set(json.loads((loc / "zarr.json").read_text())["consolidated_metadata"]["metadata"])
    dropped = ("images/s_hires_image", "tables/square_002um", "shapes/s_square_016um")
    assert not [k for k in keys if k.startswith(dropped)]
    assert any(k.startswith("tables/square_008um/") for k in keys)


def test_open_elements_missing_path_raises(vhd_store, tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        remote.open_sdata(str(vhd_store), lazy=False, elements=["tables/nope"], cache_dir=tmp_path / "c")


def test_open_elements_empty_raises(vhd_store, tmp_path):
    with pytest.raises(ValueError, match="element paths"):
        remote.open_sdata(str(vhd_store), lazy=False, elements=[], cache_dir=tmp_path / "c")


def test_open_elements_lazy_rejected(vhd_store, tmp_path):
    with pytest.raises(NotImplementedError, match="lazy=False"):
        remote.open_sdata(str(vhd_store), lazy=True, elements=KEEP, cache_dir=tmp_path / "c")


def test_prune_consolidated_noop_without_block(tmp_path):
    zj = tmp_path / "zarr.json"
    zj.write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
    before = zj.read_text()
    remote._prune_consolidated(tmp_path, {"images"})
    assert zj.read_text() == before


def test_cache_full_then_partial_no_collision(vhd_store, tmp_path):
    cache = tmp_path / "cache"
    full = remote.open_sdata(str(vhd_store), lazy=False, cache_dir=cache)
    assert "square_002um" in full.tables
    part = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=cache)
    assert set(part.tables) == {"square_008um"}
    dirs = sorted(p.name for p in cache.glob("*.zarr"))
    assert len(dirs) == 2
    assert any("__" in d for d in dirs)


def test_partial_copy_atomic_on_failure(vhd_store, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    orig = remote._prune_consolidated
    calls = {"n": 0}

    def boom(dest, kept):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated mid-build failure")
        return orig(dest, kept)

    monkeypatch.setattr(remote, "_prune_consolidated", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=cache)
    assert not list(cache.glob("vhd__*.zarr"))
    assert not list(cache.glob("vhd.partial-*"))
    monkeypatch.undo()
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=cache)
    assert set(out.tables) == {"square_008um"}


def test_prune_zmetadata_v2(tmp_path):
    import json

    from sddb.remote import _prune_consolidated

    dest = tmp_path / "v2.zarr"
    (dest / "images" / "keep" / "s0").mkdir(parents=True)
    (dest / "images" / "drop").mkdir(parents=True)
    zmeta = {
        "zarr_consolidated_format": 1,
        "metadata": {
            ".zgroup": {"zarr_format": 2},
            "images/.zgroup": {"zarr_format": 2},
            "images/keep/.zgroup": {"zarr_format": 2},
            "images/keep/s0/.zarray": {"shape": [3, 4, 4]},
            "images/drop/.zgroup": {"zarr_format": 2},
            "images/drop/.zarray": {"shape": [3, 2, 2]},
        },
    }
    (dest / ".zmetadata").write_text(json.dumps(zmeta))
    _prune_consolidated(dest, {"images", "images/keep"})
    out = json.loads((dest / ".zmetadata").read_text())["metadata"]
    assert "images/keep/s0/.zarray" in out
    assert not any(k.startswith("images/drop") for k in out)
    assert "images/.zgroup" in out
    assert ".zgroup" in out


def test_prune_zmetadata_unknown_form_rejected(tmp_path):
    import pytest

    from sddb.remote import _prune_consolidated

    dest = tmp_path / "bad.zarr"
    dest.mkdir()
    (dest / ".zmetadata").write_text('{"weird": 1}')
    with pytest.raises(NotImplementedError):
        _prune_consolidated(dest, {"images", "images/keep"})


def test_region_closure_autoincludes(vhd_store, tmp_path):
    # request ONLY the 008um table; its region shapes must be auto-included so the annotation resolves
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=["tables/square_008um"], cache_dir=tmp_path / "c")
    assert "s_square_008um" in out.shapes  # auto-included
    assert out.tables["square_008um"].obs["region"].cat.categories.tolist() == ["s_square_008um"]
