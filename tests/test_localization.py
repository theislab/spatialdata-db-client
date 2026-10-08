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
