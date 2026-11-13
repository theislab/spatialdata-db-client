"""The loader adapters (Task 3 of T144): Rasterize, Centroids, TissueMask and StainNormalize on synthetic SpatialData."""

from __future__ import annotations

import logging

import anndata as ad
import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import spatialdata as sd
from dask.callbacks import Callback
from scipy import ndimage
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity, Scale
from tests._contract_fixture import draw_cells

from sddb.contract import AdapterRecord
from sddb.loader import (
    Centroids,
    LoaderItem,
    Rasterize,
    StainNormalize,
    TissueMask,
    _adapters,
    adapter_from_record,
    default_chain,
)

RUIFROK_H = np.array([0.65, 0.70, 0.29])
RUIFROK_E = np.array([0.07, 0.99, 0.11])
TABLE_ROLES = {"image": "^images/he$", "table": "^tables/table$"}


# -- synthetic SpatialData ---------------------------------------------------------------------------------------------


def make_sdata(
    image: np.ndarray,
    *,
    image_tf=None,
    scale_factors=None,
    shapes=None,
    shapes_tf=None,
    labels: np.ndarray | None = None,
    labels_tf=None,
    ids=None,
    annotate: str = "cells",
    obs: dict | None = None,
    image_chunks: int | None = None,
) -> sd.SpatialData:
    """An in-memory SpatialData: image ``he``, optional shapes ``cells`` and labels ``cell_labels``, table ``table``.

    `shapes` is a GeoDataFrame (polygons, or points with a ``radius`` column) indexed by the instance ids. The table
    annotates `annotate` with instance key ``cell_id`` = `ids` (the shapes' index, or ``1..n`` for labels).
    """
    cs = {"global": image_tf if image_tf is not None else Identity()}
    data = da.from_array(image, chunks=(image.shape[0], image_chunks or image.shape[1], image_chunks or image.shape[2]))
    images = {"he": Image2DModel.parse(data, dims=("c", "y", "x"), scale_factors=scale_factors, transformations=cs)}
    elements: dict = {"images": images}
    if shapes is not None:
        elements["shapes"] = {
            "cells": ShapesModel.parse(shapes, transformations={"global": shapes_tf if shapes_tf else Identity()})
        }
    if labels is not None:
        elements["labels"] = {
            "cell_labels": Labels2DModel.parse(
                labels, dims=("y", "x"), transformations={"global": labels_tf if labels_tf else Identity()}
            )
        }
    if ids is None:
        ids = list(shapes.index) if annotate == "cells" else list(range(1, int(labels.max()) + 1))
    n = len(ids)
    table = ad.AnnData(
        X=np.random.default_rng(0).poisson(2.0, (n, 5)).astype(np.int32),
        obs=pd.DataFrame(
            {"cell_id": ids, "region": pd.Categorical([annotate] * n), **(obs or {})}, index=[f"r{i}" for i in range(n)]
        ),
    )
    elements["tables"] = {
        "table": TableModel.parse(table, region=annotate, region_key="region", instance_key="cell_id")
    }
    return sd.SpatialData(**elements)


def item_of(sdata, *, role="he", mpp_native=0.25, roles=None, **kwargs) -> LoaderItem:
    return LoaderItem.from_sdata(
        sdata,
        uid="uid0001",
        study_id="Smith2023",
        role=role,
        roles=roles or TABLE_ROLES,
        mpp_native=mpp_native,
        **kwargs,
    )


def squares(centres, side) -> gpd.GeoDataFrame:
    """Axis-aligned squares (one per ``(x, y)`` centre) indexed by string ids."""
    geoms = [shapely.box(x - side / 2, y - side / 2, x + side / 2, y + side / 2) for x, y in centres]
    return gpd.GeoDataFrame(geometry=geoms, index=[f"cell-{i}" for i in range(len(geoms))])


def he_image(height, width, *, seed=0, disk=None) -> np.ndarray:
    """An H&E-like uint8 image (Beer-Lambert with Ruifrok vectors): tissue in `disk` ``(cy, cx, r)``, glass outside."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    inside = np.ones((height, width), bool) if disk is None else np.hypot(yy - disk[0], xx - disk[1]) <= disk[2]
    c_h = rng.uniform(0.2, 1.5, (height, width)) * inside
    c_e = rng.uniform(0.1, 1.0, (height, width)) * inside
    od = c_h[..., None] * RUIFROK_H / np.linalg.norm(RUIFROK_H) + c_e[..., None] * RUIFROK_E / np.linalg.norm(RUIFROK_E)
    rgb = 250.0 * np.exp(-od) + rng.normal(0, 1.5, (height, width, 3))
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8).transpose(2, 0, 1)


# -- Rasterize ---------------------------------------------------------------------------------------------------------


def test_rasterize_refuses_upsampling():
    img = np.zeros((3, 32, 32), np.uint8)
    sdata = make_sdata(img, shapes=squares([(10, 10)], 4))
    with pytest.raises(ValueError, match=r"0\.505.*0\.5|0\.5.*0\.505"):
        Rasterize(0.5).build(item_of(sdata, mpp_native=0.505))
    item = Rasterize(0.5, allow_upsample=True).build(item_of(sdata, mpp_native=0.505))
    assert item.flags["upsampled"]
    assert item.flags["allow_upsample"]
    assert item.grid == "target"
    assert item.image.shape == (3, 33, 33)


def test_rasterize_downsample_matches_reference():
    img = np.random.default_rng(1).integers(0, 256, (3, 64, 64), dtype=np.uint8)
    sdata = make_sdata(img, shapes=squares([(20, 20)], 8), image_chunks=16)
    item = Rasterize(0.5, tile=16).build(item_of(sdata, mpp_native=0.25))
    blocks = img.reshape(3, 32, 2, 32, 2).astype(np.float64).mean(axis=(2, 4))
    np.testing.assert_array_equal(item.image.values, np.floor(blocks + 0.5).astype(np.uint8))
    assert isinstance(item.image.data, da.Array)
    assert item.image.dtype == np.uint8
    assert item.grid == "target"
    assert item.mpp_target == 0.5
    assert item.mpp_native == 0.25
    assert not item.flags["upsampled"]
    assert item.labels["cells"].shape == (32, 32)
    assert item.records == [
        AdapterRecord(name="Rasterize", version="0.1.0", params={"mpp": 0.5, "allow_upsample": False, "tile": 16})
    ]


def test_rasterize_uses_coarsest_level_not_finer_than_target():
    native = 0.25
    shapes = [(64, 64), (32, 32), (16, 16)]  # scale0, scale1 (0.5 um), scale2 (1.0 um)
    assert _adapters._pick_level(shapes, native, 0.6) == 1
    assert _adapters._pick_level(shapes, native, 0.5) == 1
    assert _adapters._pick_level(shapes, native, 0.49) == 0
    assert _adapters._pick_level(shapes, native, 2.0) == 2


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_rasterize_bakes_rotation(scale):
    img = np.random.default_rng(2).integers(0, 256, (3, 40, 60), dtype=np.uint8)
    height, width = img.shape[1:]
    # x_cs = s * y, y_cs = s * (W - x): a 90 degree rotation (plus scale), like the Atera H&E onto its global frame
    tf = Affine(
        np.array([[0, scale, 0], [-scale, 0, scale * width], [0, 0, 1]]), input_axes=("x", "y"), output_axes=("x", "y")
    )
    sdata = make_sdata(img, image_tf=tf, shapes=squares([(10, 10)], 4), image_chunks=16)
    mpp_native = 0.5
    item = Rasterize(mpp_native if scale != 1 else None, tile=16).build(item_of(sdata, mpp_native=mpp_native))
    np.testing.assert_array_equal(item.image.values, np.rot90(img, axes=(1, 2)))
    assert item.grid == ("target" if scale != 1 else "native")


def test_rasterize_without_mpp_native_keeps_native_grid():
    img = np.zeros((3, 20, 30), np.uint8)
    sdata = make_sdata(img, shapes=squares([(10, 10)], 4))
    item = Rasterize(0.5).build(item_of(sdata, mpp_native=None))
    assert item.grid == "native"
    assert item.mpp_target is None
    assert item.image.shape == (3, 20, 30)


def test_polygons_rasterized_once_with_instance_ids(monkeypatch):
    rng = np.random.default_rng(3)
    centres = [(x + rng.uniform(-1, 1), y + rng.uniform(-1, 1)) for y in range(6, 64, 13) for x in range(6, 64, 13)]
    geoms = [shapely.Point(c).buffer(4.3, quad_segs=3).simplify(0.01) for c in centres]  # hexagon-ish polygons
    cells = gpd.GeoDataFrame(geometry=geoms, index=[f"cell-{i}" for i in range(len(geoms))])
    img = np.zeros((3, 64, 64), np.uint8)
    sdata = make_sdata(img, shapes=cells.iloc[::-1])  # table rows in the shapes' order, reversed vs. drawing order

    calls = []
    draw = _adapters._draw_tile
    monkeypatch.setattr(_adapters, "_draw_tile", lambda *a, **k: calls.append(1) or draw(*a, **k))
    item = Rasterize(None, tile=16).build(item_of(sdata))
    labels = item.labels["cells"].values
    assert len(calls) == 16  # one draw per 16 x 16 tile, all cells at once
    assert labels.dtype == np.uint32

    # the reference: each pixel centre takes the label id of the polygon containing it
    ids = dict(zip(item.table.obs["cell_id"], item.table.obs["label_id"], strict=True))
    yy, xx = np.mgrid[:64, :64] + 0.5
    expected = np.zeros((64, 64), np.uint32)
    for name, geom in zip(cells.index, cells.geometry, strict=True):
        expected[shapely.contains_xy(geom, xx, yy)] = ids[name]
    np.testing.assert_array_equal(labels, expected)
    assert set(np.unique(labels)) - {0} == set(item.table.obs["label_id"])


def test_circles_filled():
    centre, radius = np.array([[20.3, 30.7]]), 6.0
    circles = gpd.GeoDataFrame({"radius": [radius]}, geometry=[shapely.Point(centre[0])], index=["c0"])
    img = np.zeros((3, 128, 96), np.uint8)
    sdata = make_sdata(img, shapes=circles, shapes_tf=Scale([2.0, 2.0], axes=("x", "y")))
    labels = Rasterize(None, tile=32).build(item_of(sdata)).labels["cells"].values
    yy, xx = np.mgrid[:128, :96] + 0.5
    dist = np.hypot(xx - 2 * centre[0, 0], yy - 2 * centre[0, 1])
    r = 2 * radius
    assert (labels[dist < r - 0.5] == 1).all()  # filled, not a ring
    assert (labels[dist > r + 0.5] == 0).all()
    assert abs((labels == 1).sum() / (np.pi * r**2) - 1) < 0.03


def test_existing_labels_resampled_nearest():
    labels = draw_cells(30, 80, 100)
    img = np.zeros((3, 80, 100), np.uint8)
    sdata = make_sdata(img, labels=labels, annotate="cell_labels")
    item = Rasterize(0.5, tile=16).build(
        item_of(sdata, roles={**TABLE_ROLES, "labels": "^labels/cell_labels$"}, mpp_native=0.25)
    )
    out = item.labels["cells"]
    assert isinstance(out.data, da.Array)
    assert out.dtype == labels.dtype
    # order 0 at factor 2: written pixel j samples source position 2j + 0.5, i.e. pixel 2j or 2j + 1, never a blend
    sub = out.values
    assert set(np.unique(sub)) <= set(np.unique(labels))
    assert ((sub == labels[::2, ::2]) | (sub == labels[1::2, 1::2])).mean() > 0.97


def test_existing_labels_need_a_row_mapping():
    labels = draw_cells(4, 40, 40)

    def shapes():
        out = squares([(10, 10), (30, 10), (10, 30), (30, 30)], 6)
        out.index = [123456789, 223456789, 323456789, 423456789]  # packed integer cell ids, not the label values
        return out

    sdata = make_sdata(np.zeros((3, 40, 40), np.uint8), shapes=shapes(), labels=labels)
    roles = {**TABLE_ROLES, "labels": "^labels/cell_labels$"}
    with pytest.raises(ValueError, match="label_column"):
        item_of(sdata, roles=roles)
    # the Atera layout: obs["label_id"] holds each row's value in the labels raster
    sdata = make_sdata(np.zeros((3, 40, 40), np.uint8), shapes=shapes(), labels=labels, obs={"label_id": [4, 3, 2, 1]})
    item = item_of(sdata, roles=roles)
    assert item.label_rule == "obs['label_id']"
    assert item.table.obs["label_id"].tolist() == [4, 3, 2, 1]


# -- Centroids ---------------------------------------------------------------------------------------------------------


def test_centroids_equal_scipy_center_of_mass():
    labels = draw_cells(30, 90, 100)
    img = np.zeros((3, 90, 100), np.uint8)
    order = np.random.default_rng(4).permutation(30) + 1  # table rows in a different order than the label values
    sdata = make_sdata(img, labels=labels, annotate="cell_labels", ids=order.tolist())
    item = Rasterize(None, tile=32).build(item_of(sdata, roles={**TABLE_ROLES, "labels": "^labels/cell_labels$"}))
    item = Centroids().build(item)
    com = np.asarray(ndimage.center_of_mass(np.ones_like(labels), labels, order))[:, ::-1]
    np.testing.assert_allclose(item.table.obsm["spatial"], com, atol=1e-9)
    assert item.table.obsm["spatial"].dtype == np.float64
    assert item.table.obs["label_id"].tolist() == order.tolist()
    assert item.records[-1] == AdapterRecord(name="Centroids", version="0.1.0", params={})


def test_centroids_tiny_cell_dropped_and_counted(caplog):
    cells = squares([(8, 8), (24, 8), (8, 24)], 6)
    tiny = gpd.GeoDataFrame(geometry=[shapely.box(24.6, 24.6, 24.8, 24.8)], index=["tiny"])  # covers no pixel centre
    sdata = make_sdata(np.zeros((3, 32, 32), np.uint8), shapes=pd.concat([cells, tiny]))
    item = Rasterize(None, tile=16).build(item_of(sdata))
    with caplog.at_level(logging.WARNING, logger="sddb.loader"):
        item = Centroids().build(item)
    assert item.table.n_obs == 3
    assert "tiny" not in item.table.obs["cell_id"].tolist()
    assert len(item.table.obsm["spatial"]) == 3
    assert "dropped 1 of 4 rows" in caplog.text


# -- TissueMask --------------------------------------------------------------------------------------------------------


def test_tissue_mask_flags_cells_outside():
    img = he_image(128, 128, disk=(64, 64, 40))
    inside = [(64, 64), (50, 70), (80, 60)]
    outside = [(10, 10), (118, 10), (10, 118), (118, 118)]
    sdata = make_sdata(img, shapes=squares(inside + outside, 5), image_chunks=32)
    item = Rasterize(None, tile=32).build(item_of(sdata))
    item = TissueMask(auto_max_pixels=1024).build(Centroids().build(item))
    tissue = item.labels["tissue"]
    assert isinstance(tissue.data, da.Array)
    assert tissue.shape == (128, 128)
    assert tissue.dtype == np.uint8
    assert item.table.obs["in_tissue"].dtype == bool
    assert item.table.obs["in_tissue"].tolist() == [True] * 3 + [False] * 4
    assert item.records[-1].name == "TissueMask"
    assert item.records[-1].params == {"auto_max_pixels": 1024}


def test_tissue_mask_needs_centroids():
    sdata = make_sdata(he_image(32, 32), shapes=squares([(16, 16)], 5))
    with pytest.raises(ValueError, match="Centroids"):
        TissueMask().build(Rasterize(None).build(item_of(sdata)))


# -- StainNormalize ----------------------------------------------------------------------------------------------------


def _he_item(seed, *, tile=32, size=128):
    img = he_image(size, size, seed=seed, disk=(size / 2, size / 2, size * 0.35))
    sdata = make_sdata(img, shapes=squares([(size / 2, size / 2), (10, 10)], 5), image_chunks=tile)
    item = Rasterize(None, tile=tile).build(item_of(sdata))
    return TissueMask(auto_max_pixels=1024).build(Centroids().build(item))


def test_stain_pooled_fit_one_reference(monkeypatch):
    import squidpy.experimental.im as sqim

    calls = []
    fit = sqim.fit_stain_reference
    monkeypatch.setattr(sqim, "fit_stain_reference", lambda *a, **k: calls.append(k) or fit(*a, **k))

    items = [_he_item(0), _he_item(1)]
    step = StainNormalize(fit_pixels=1024)
    reference = step.fit(items)
    assert len(calls) == 1
    assert len(calls[0]["image_key"]) == 2
    assert step.reference is reference

    out = [step.build(item) for item in items]
    for item in out:
        assert isinstance(item.image.data, da.Array)
        assert item.image.dtype == np.uint8
        assert item.image.shape == (3, 128, 128)
        assert set(item.stain_reference) == {"method", "stain_matrix", "white_point", "max_concentrations"}
        assert np.asarray(item.stain_reference["stain_matrix"]).shape == (3, 3)
    assert out[0].stain_reference == out[1].stain_reference
    assert out[0].records[-1] == out[1].records[-1]
    np.testing.assert_allclose(out[0].stain_reference["stain_matrix"], reference.stain_matrix)


def test_stain_refuses_if():
    img = np.random.default_rng(5).integers(0, 2**16, (2, 32, 32), dtype=np.uint16)
    sdata = make_sdata(img, shapes=squares([(16, 16)], 5))
    item = Rasterize(None).build(item_of(sdata, role="if"))
    assert item.image.dtype == np.uint16
    assert item.channels == ["0", "1"]
    with pytest.raises(ValueError, match="H&E"):
        StainNormalize().fit([item])
    reference = StainNormalize(fit_pixels=1024).fit([_he_item(0)])
    with pytest.raises(ValueError, match="H&E"):
        StainNormalize(reference=reference).build(item)


def test_stain_build_without_reference_raises():
    with pytest.raises(ValueError, match="fit"):
        StainNormalize().build(_he_item(0))


# -- records, dtype, protocol ------------------------------------------------------------------------------------------


def test_adapter_record_roundtrip():
    reference = StainNormalize(fit_pixels=1024).fit([_he_item(0)])
    for step in [
        Rasterize(0.5, allow_upsample=True, tile=256),
        Rasterize(None),
        Centroids(),
        TissueMask(method="otsu", auto_max_pixels=1000),
        StainNormalize(fit_pixels=4096),
        StainNormalize(reference=reference),
    ]:
        rec = step.to_record()
        back = type(step).from_record(rec)
        assert back.to_record() == rec
        assert adapter_from_record(rec).to_record() == rec
        assert AdapterRecord.model_validate_json(rec.model_dump_json()) == rec  # params are plain JSON
    back = StainNormalize.from_record(StainNormalize(reference=reference).to_record())
    np.testing.assert_allclose(back.reference.stain_matrix, reference.stain_matrix)
    with pytest.raises(ValueError, match="Centroids"):
        Rasterize.from_record(Centroids().to_record())


def test_adapters_satisfy_task_adapter_protocol():
    from sddb.adapter import TaskAdapter

    for step in default_chain(0.5):
        assert isinstance(step, TaskAdapter)
    assert [type(s).__name__ for s in default_chain(0.5)] == ["Rasterize", "Centroids", "TissueMask", "StainNormalize"]


def test_he_dtype_rule_recorded():
    img = (np.random.default_rng(6).integers(0, 2**16, (3, 16, 16))).astype(np.uint16)
    sdata = make_sdata(img, shapes=squares([(8, 8)], 4))
    item = Rasterize(None).build(item_of(sdata))
    assert item.image.dtype == np.uint8
    assert item.dtype_rule is not None
    assert "uint16" in item.dtype_rule
    np.testing.assert_array_equal(item.image.values, np.floor(img / 257 + 0.5).astype(np.uint8))


# -- the dask guard ----------------------------------------------------------------------------------------------------


class _NoWholeImage(Callback):
    """Fails if any dask task returns an array with at least `max_px` pixels in its last two dimensions."""

    def __init__(self, max_px: int) -> None:
        super().__init__()
        self.max_px = max_px
        self.largest = 0

    def _posttask(self, key, result, dsk, state, worker_id):
        if isinstance(result, np.ndarray) and result.ndim >= 2:
            px = result.shape[-1] * result.shape[-2]
            self.largest = max(self.largest, px)
            assert px < self.max_px, f"task {key} computed {result.shape}, a whole image"


def _atera_shaped(tile):
    """A rotated, scaled H&E onto `global` (the Atera layout), cell labels on `global` and `label_id` in obs."""
    s, h0, w0 = 1.25, 400, 480  # H&E 0.25 um/px onto a 0.2 um global grid
    tf = Affine(np.array([[0, s, 0], [-s, 0, s * w0], [0, 0, 1]]), input_axes=("x", "y"), output_axes=("x", "y"))
    img = he_image(h0, w0, disk=(200, 240, 170))
    labels = draw_cells(150, int(s * w0), int(s * h0))
    n = int(labels.max())
    shapes = squares([(1, 1)] * n, 1)
    shapes.index = np.arange(10**8, 10**8 + n)
    return make_sdata(
        img, image_tf=tf, shapes=shapes, labels=labels, image_chunks=tile, obs={"label_id": np.arange(1, n + 1)}
    )


def test_chain_never_computes_whole_image():
    tile = 64
    sdata = _atera_shaped(tile)
    roles = {**TABLE_ROLES, "labels": "^labels/cell_labels$"}
    items = [item_of(sdata, roles=roles, mpp_native=0.25)]
    rasterize, centroids, tissue, stain = default_chain(0.5)
    rasterize.tile, tissue.config["auto_max_pixels"], stain.fit_pixels = tile, 1024, 1024
    written = (240, 200)  # the global footprint 600 x 500 at 2.5 global units per written pixel
    guard = _NoWholeImage(max_px=written[0] * written[1] // 2)
    with dask.config.set(scheduler="sync"), guard:
        items = [tissue.build(centroids.build(rasterize.build(item))) for item in items]
        stain.fit(items)
        item = stain.build(items[0])
        image = np.empty(item.image.shape, item.image.dtype)
        cells = np.empty(item.labels["cells"].shape, item.labels["cells"].dtype)
        da.store([item.image.data, item.labels["cells"].data], [image, cells], lock=False)  # block by block
    assert item.image.shape == (3, *written)
    assert cells.shape == written
    assert all(isinstance(a.data, da.Array) for a in (item.image, item.labels["cells"], item.labels["tissue"]))
    assert 0 < guard.largest < guard.max_px

    # centroids lie in their own label (R2) and inside the image
    xy = item.table.obsm["spatial"]
    px = np.floor(xy + 0.5).astype(int)
    assert ((px >= 0) & (px < [written[1], written[0]])).all()
    assert (cells[px[:, 1], px[:, 0]] == item.table.obs["label_id"].to_numpy()).mean() >= 0.99

    # the guard does trip on a whole-image compute
    with (
        dask.config.set(scheduler="sync"),
        pytest.raises(AssertionError, match="whole image"),
        _NoWholeImage(guard.max_px),
    ):
        item.image.data.compute()
