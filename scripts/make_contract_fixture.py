"""Build the loader-contract fixture: a crop of the Atera breast (FF) H&E with its cells, run through ``materialize``.

Usage, on a compute node (about 8 GB RAM for the read of the full bundle)::

    python scripts/make_contract_fixture.py <atera bundle dir> <out dir> [--side-um 400]

Writes, under <out dir>:

- ``overview.png``: the cell-density overview the window is chosen from, and the window on the coarsest H&E level;
- ``atera-breast-crop.zarr``: the crop as a SpatialData (``he_image``, ``cell_boundaries``, ``table``);
- ``sddb-contract-0.1.0-atera-breast-crop.zarr`` (+ ``.zarr.sddb`` run state) and its ``.zarr.zip``;
- ``overlay.png``: the written H&E with the cell outlines and centroids (spatialdata_plot);
- ``summary.json``: what the report needs (window, mpp, counts, shapes, validate, Loader batch shapes).

Sources of the numbers (checked 2026-11-12):

- microns per morphology pixel: ``experiment.spatial`` ``"pixel_size"`` of the bundle (0.21249222 for this bundle);
  the vendor spec registry (theislab/spatialdata-db-schemas specs/10x_atera.yaml at 0562c6c5f) gives the same value
  and says to prefer the per-dataset one. The ``global`` coordinate system of the reader is in morphology pixels
  (spatialdata-io atera.py ``_get_aligned_images``: "in the global (morphology px) coordinate system").
- ``obs["centroid_column"/"centroid_row"]`` are in microns: checked below against the cell boundaries' centroids.
- mpp of the H&E: (global units per H&E pixel) x pixel_size, the global units per pixel being
  ``sqrt(|det A|)`` of the H&E's 2x2 affine onto ``global``, the same scale ``Rasterize`` uses.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import zipfile
from pathlib import Path

import numpy as np

# The crop window, chosen by the rule in `choose_window` from the bundle below; the run checks it again.
BUNDLE_NAME = "Atera_v1_WTA_S4C_Breast_Cancer_FF"
# the released record of this bundle (candidate sddb_a722jfj3pq, same canonical_key); Catalog().release is "2026-11-12"
UID = "Oc0RVwT6WdZXOkAv0000"
STUDY_ID = "10x:atera-wta-ff-human-breast-cancer"
CATALOG_RELEASE = "2026-11-12"
SIDE_UM = 400.0  # 1000 um gave 83.5 MB zipped (X 88 of 105 MB on disk)
BIN_UM = 100.0
# (x0, y0, x1, y1) in microns of the global frame; the densest 1000 um window was (4800, 1600, 5800, 2600)
WINDOW_UM: tuple[float, float, float, float] | None = (5400.0, 2000.0, 5800.0, 2400.0)
FIXTURE = "sddb-contract-0.1.0-atera-breast-crop.zarr"
ROLES = {"image": "^images/he_image$", "table": "^tables/table$"}
MPP_TARGET = 0.5


def choose_window(cx_um: np.ndarray, cy_um: np.ndarray, side_um: float, bin_um: float) -> tuple[np.ndarray, tuple]:
    """The window of `side_um` with the most cell centroids, on a grid of `bin_um` bins over the centroids.

    Bins start at the lowest centroid (floored to a bin); windows are whole bins; the first maximum (row-major over
    (y, x)) wins. Returns the density map and the window ``(x0, y0, x1, y1)`` in microns.
    """
    x0, y0 = math.floor(cx_um.min() / bin_um) * bin_um, math.floor(cy_um.min() / bin_um) * bin_um
    nx = math.ceil((cx_um.max() - x0) / bin_um) + 1
    ny = math.ceil((cy_um.max() - y0) / bin_um) + 1
    counts, _, _ = np.histogram2d(cy_um, cx_um, bins=(ny, nx), range=((y0, y0 + ny * bin_um), (x0, x0 + nx * bin_um)))
    k = round(side_um / bin_um)
    c = np.pad(counts.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    iy, ix = np.unravel_index(int(np.argmax(sums)), sums.shape)
    wx0, wy0 = x0 + ix * bin_um, y0 + iy * bin_um
    return counts, (float(wx0), float(wy0), float(wx0 + k * bin_um), float(wy0 + k * bin_um))


class Member:
    """The part of :class:`sddb.Dataset` that ``materialize`` reads, as in tests/test_loader_materialize.py."""

    def __init__(self, uid: str, study_id: str, path: Path):
        self.uid, self.study_id, self._path = uid, study_id, path

    def to_sdata(self):
        """Open the crop lazily."""
        import spatialdata as sd

        return sd.read_zarr(self._path)


def main() -> None:
    """Read, choose the window, crop, materialize, validate, zip, plot and load."""
    import spatialdata as sd
    import spatialdata_io
    from scipy import sparse
    from spatialdata import bounding_box_query
    from spatialdata.transformations import get_transformation

    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--side-um", type=float, default=SIDE_UM)
    args = p.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"bundle": str(args.bundle)}
    t0 = time.time()

    spec = json.loads((args.bundle / "experiment.spatial").read_text())
    ps = float(spec["pixel_size"])
    summary["experiment.spatial"] = {k: spec[k] for k in ("pixel_size", "region_name", "sample_type", "num_cells")}

    sdata = spatialdata_io.atera(
        args.bundle,
        aligned_images=True,
        morphology_images=True,
        cells_labels=False,  # the table then annotates cell_boundaries (by cell_id): the cells come from the shapes
        nucleus_labels=False,
        nucleus_boundaries=False,
    )
    print(sdata, flush=True)
    assert float(sdata.attrs["pixel_size"]) == ps, (sdata.attrs["pixel_size"], ps)
    table = sdata.tables["table"]
    summary["source"] = {
        "elements": [f"{k}/{n}" for k, n, _ in sdata.gen_elements()],
        "table": list(table.shape),
        "he_scales": {k: list(v["image"].shape) for k, v in sdata.images["he_image"].items()},
        "read_s": round(time.time() - t0, 1),
    }

    # obs centroids are in microns: compare with the boundaries' centroids in global px x pixel_size
    obs = table.obs
    cx, cy = obs["centroid_column"].to_numpy(float), obs["centroid_row"].to_numpy(float)
    shapes = sdata.shapes["cell_boundaries"]
    sample = obs["cell_id"].to_numpy()[:2000]
    g = sd.transform(shapes.loc[sample], to_coordinate_system="global").geometry.centroid
    diff_um = np.hypot(g.x.to_numpy() * ps - cx[:2000], g.y.to_numpy() * ps - cy[:2000])
    summary["centroid_check_median_um"] = float(np.median(diff_um))
    assert np.median(diff_um) < 1.0, np.median(diff_um)

    he = sdata.images["he_image"]
    a = get_transformation(he, "global").to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    he_scale = math.sqrt(abs(np.linalg.det(np.asarray(a)[:2, :2])))  # global (morphology px) per H&E px
    summary["he_affine"] = np.asarray(a).tolist()
    summary["he_global_per_px"] = he_scale
    summary["mpp_native"] = he_scale * ps

    density, window = choose_window(cx, cy, args.side_um, BIN_UM)
    summary["window_um"] = window
    if WINDOW_UM is not None and args.side_um == SIDE_UM:
        assert np.allclose(window, WINDOW_UM), (window, WINDOW_UM)
    inside = (cx >= window[0]) & (cx < window[2]) & (cy >= window[1]) & (cy < window[3])
    summary["window_cells_by_centroid"] = int(inside.sum())
    summary["window_nnz_estimate"] = int(obs["n_genes"].to_numpy()[inside].sum())
    print(json.dumps(summary, indent=1, default=str), flush=True)
    _overview(he, a, density, window, ps, out / "overview.png")

    # io atera: obs index name duplicates the barcode column; fixed upstream in the fork, remove this once the env pins
    # that commit.
    # The reader keeps obs' index ("barcode") as a column of the same name too (spatialdata-io atera.py
    # _align_index_column); spatialdata's table join (relational_query.py:259, obs.reset_index()) then raises
    # "cannot insert barcode, already exists". The column keeps the values, so the index name is dropped.
    if table.obs.index.name in table.obs.columns:
        table.obs.index.name = None
    lo = np.array(window[:2]) / ps
    hi = np.array(window[2:]) / ps
    crop = bounding_box_query(
        sdata.subset(["he_image", "cell_boundaries", "table"]),
        axes=("x", "y"),
        min_coordinate=lo,
        max_coordinate=hi,
        target_coordinate_system="global",
        filter_table=True,
    )
    ct = crop.tables["table"]
    x = ct.X.compute() if hasattr(ct.X, "compute") else ct.X
    ct.X = sparse.csr_matrix(x)
    crop_path = out / "atera-breast-crop.zarr"
    crop.write(crop_path, overwrite=True)
    a_crop = get_transformation(crop.images["he_image"], "global").to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    summary["crop"] = {
        "table": list(ct.shape),
        "shapes": len(crop.shapes["cell_boundaries"]),
        "he_scale0": list(crop.images["he_image"]["scale0"]["image"].shape),
        "he_global_per_px": math.sqrt(abs(np.linalg.det(np.asarray(a_crop)[:2, :2]))),
        "nnz": int(ct.X.nnz),
    }
    print(json.dumps(summary["crop"]), flush=True)
    del sdata, crop, table, obs, shapes

    from sddb.contract import validate
    from sddb.loader import Centroids, Rasterize, materialize

    dest = out / FIXTURE
    # No TissueMask and no StainNormalize: a 400 um crop of tissue has no glass, so squidpy's otsu tissue split is
    # meaningless (it flagged 59.5% of the cells in tissue), and StainNormalize needs that mask (labels["tissue"]).
    # Both are optional in the contract; this fixture tests the contract and the loader, and the full chain is checked
    # on whole Atera slides (docs/use_cases/loader_atera.md).
    chain = [Rasterize(MPP_TARGET), Centroids()]
    # a member list has no catalog; carry the release the way a Cohort does (materialize._release reads _source)
    cohort = type("FixtureCohort", (list,), {"_source": type("Release", (), {"release": CATALOG_RELEASE})})
    materialize(
        cohort([Member(UID, STUDY_ID, crop_path)]),
        dest,
        adapters=chain,
        role="he",
        roles=ROLES,
        mpp_native={UID: summary["mpp_native"]},
    )
    summary["validate"] = validate(dest)
    summary.update(_describe(dest))
    summary["zip_bytes"] = _zip(dest, out / f"{FIXTURE}.zip")
    summary["dir_bytes"] = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(json.dumps({k: summary[k] for k in ("validate", "zip_bytes", "dir_bytes")}), flush=True)
    _overlay(dest, out / "overlay.png")
    summary["loader"] = _smoke(dest)
    summary["total_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


def _overview(he, a, density, window, ps, path: Path) -> None:
    """The density map with the window, and the coarsest H&E level with the window mapped into its pixels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    level = list(he.keys())[-1]
    img = np.asarray(he[level]["image"].transpose("y", "x", "c").data.compute())
    f = he["scale0"]["image"].shape[-1] / img.shape[1]
    x0, y0, x1, y1 = window
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]) / ps  # global px
    inv = np.linalg.inv(np.asarray(a))
    hc = (inv @ np.c_[corners, np.ones(5)].T)[:2].T / f
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(density, origin="upper", cmap="magma")
    ax[0].set_title(f"cells per {BIN_UM:g} um bin")
    ax[1].imshow(img)
    ax[1].plot(hc[:, 0], hc[:, 1], c="cyan", lw=1.5)
    ax[1].set_title(f"H&E {level}, window {x1 - x0:g} um")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _describe(dest: Path) -> dict:
    import zarr
    from anndata.io import read_elem

    g = zarr.open_group(dest / "dataset_0", mode="r")
    he = g["images"]["he"]
    contract = read_elem(g["uns"]["sddb_contract"])
    return {
        "n_cells": int(g["obsm"]["spatial"].shape[0]),
        "n_genes": int(len(read_elem(g["var"][g["var"].attrs["_index"]]))),
        "image": {"shape": list(he.shape), "dtype": str(he.dtype)},
        "labels": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in g["labels"].arrays()},
        "contract": contract,
        "in_tissue_share": float(np.asarray(read_elem(g["obs"]["in_tissue"])).mean())
        if "in_tissue" in g["obs"]
        else None,
    }


def _zip(dest: Path, path: Path) -> int:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(dest.rglob("*")):
            if f.is_file():
                z.write(f, Path(dest.name) / f.relative_to(dest))
    return path.stat().st_size


def _overlay(dest: Path, path: Path) -> None:
    """The written H&E, the cell outlines and the centroids, with spatialdata_plot; a full view and a zoom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import spatialdata as sd
    import spatialdata_plot  # noqa: F401
    import zarr
    from spatialdata.models import Image2DModel, Labels2DModel, PointsModel

    g = zarr.open_group(dest / "dataset_0", mode="r")
    xy = np.asarray(g["obsm"]["spatial"][...])
    # Centroids are pixel-index means; spatialdata puts pixel j's centre at j + 0.5
    pts = pd.DataFrame({"x": xy[:, 0] + 0.5, "y": xy[:, 1] + 0.5})
    s = sd.SpatialData(
        images={
            "he": Image2DModel.parse(np.asarray(g["images"]["he"][...]), dims=("c", "y", "x"), c_coords=["r", "g", "b"])
        },
        labels={"cells": Labels2DModel.parse(np.asarray(g["labels"]["cells"][...]), dims=("y", "x"))},
        points={"centroids": PointsModel.parse(pts)},
    )
    h, w = g["images"]["he"].shape[1:]
    z = 100  # half side of the zoom, px
    zoom = s.query.bounding_box(
        axes=("x", "y"),
        min_coordinate=[w / 2 - z, h / 2 - z],
        max_coordinate=[w / 2 + z, h / 2 + z],
        target_coordinate_system="global",
    )
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, view, size in ((axs[0], s, 0.5), (axs[1], zoom, 12)):
        (
            view.pl.render_images("he")
            .pl.render_labels("cells", outline=True, outline_color="lime", fill_alpha=0.0, contour_px=2)
            .pl.render_points("centroids", color="red", size=size, method="matplotlib")
            .pl.show(ax=ax, title="", colorbar=False)
        )
    axs[0].set_title(f"fixture, {w} x {h} px at {MPP_TARGET} um/px")
    axs[1].set_title(f"centre, {2 * z} x {2 * z} px")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _smoke(dest: Path) -> dict:
    """A few batches of the spatialbatch Loader over the fixture; the shapes of one batch."""
    from spatialbatch import Loader, SpatialDatasetCollection

    loader = Loader(images=["he"], labels=["cells"], crop_px=224, batch_size=16).use_collection(
        SpatialDatasetCollection(dest)
    )
    out: dict = {"batches": 0}
    for i, batch in enumerate(loader):
        if i == 0:
            out["first"] = {
                k: (
                    {n: [list(v.shape), str(v.dtype)] for n, v in batch[k].items()}
                    if isinstance(batch[k], dict)
                    else [list(getattr(batch[k], "shape", ())), type(batch[k]).__name__]
                )
                for k in batch
            }
        out["batches"] = i + 1
        if i == 4:
            break
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
