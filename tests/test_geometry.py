import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale

from sddb.geometry import table_centroids

CENTRES = np.array([[4.0, 4.0], [20.0, 4.0], [4.0, 20.0]])


def _shapes(name_ids=(0, 1, 2)):
    geoms = [box(cx - 4, cy - 4, cx + 4, cy + 4) for cx, cy in CENTRES]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, index=list(name_ids))
    return ShapesModel.parse(gdf, transformations={"global": Identity()})


def _image(transformations=None):
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    return Image2DModel.parse(arr, dims=("c", "y", "x"), transformations=transformations or {"global": Identity()})


def _table(regions, ids, region_names=None):
    obs = pd.DataFrame(
        {"region": pd.Categorical(region_names or [regions[0]] * len(ids)), "location_id": list(ids)},
        index=[f"r{i}" for i in range(len(ids))],
    )
    adata = ad.AnnData(X=np.zeros((len(ids), 1)), obs=obs)
    return TableModel.parse(adata, region=list(regions), region_key="region", instance_key="location_id")


def _sdata(image_tf=None, ids=(0, 1, 2)):
    return SpatialData(
        images={"img": _image(image_tf)},
        shapes={"sq": _shapes()},
        tables={"t": _table(["sq"], ids)},
    )


def test_identity_transform():
    out = table_centroids(_sdata(), "t", "img")
    assert out.shape == (3, 2)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, CENTRES)


def test_image_scaled():
    sd = _sdata({"global": Scale([2, 2], axes=("y", "x"))})
    np.testing.assert_allclose(table_centroids(sd, "t", "img"), CENTRES / 2)


def test_table_order_follows_instance_key():
    ids = (2, 0, 1)
    out = table_centroids(_sdata(ids=ids), "t", "img")
    np.testing.assert_allclose(out, CENTRES[list(ids)])


def test_no_shared_coordinate_system_raises():
    sd = _sdata({"other": Identity()})
    with pytest.raises(ValueError, match="img.*sq|sq.*img"):
        table_centroids(sd, "t", "img")


def test_multi_region_table_raises():
    sd = SpatialData(
        images={"img": _image()},
        shapes={"sq": _shapes(), "sq2": _shapes()},
        tables={"t": _table(["sq", "sq2"], [0, 1], ["sq", "sq2"])},
    )
    with pytest.raises(ValueError, match="one region|single|multiple"):
        table_centroids(sd, "t", "img")
