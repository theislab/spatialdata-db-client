"""Materialize whole Atera H&E slides into a loader collection on Slurm, and check the result.

Three steps, each run on a compute node (see ``docs/use_cases/loader_atera.md``)::

    python scripts/atera_loader_cohort.py store <atera bundle dir> <uid> <stores dir>
    python scripts/atera_loader_cohort.py materialize <stores dir> <dest> [--workers 4] [--slurm-json '{...}']
    python scripts/atera_loader_cohort.py check <stores dir> <dest> <out dir>

``store`` reads one bundle with ``spatialdata_io.atera`` and writes the H&E, the cell boundaries and the table as a
SpatialData store, plus ``<uid>.json`` with the slide's numbers. ``materialize`` builds the cohort from the stores (one
member per slide) and runs ``sddb.loader.materialize`` with the snakemake runner on Slurm. ``check`` validates the
collection, counts the cells in tissue, measures what StainNormalize changed and plots the tissue masks.

Sources of the numbers (checked 2026-11-12):

- microns per morphology pixel: the bundle's ``experiment.spatial`` ``"pixel_size"``; the vendor spec registry
  (theislab/spatialdata-db-schemas specs/10x_atera.yaml at 0562c6c5f) says to prefer the per-dataset value. The
  ``global`` coordinate system of the reader is in morphology pixels (spatialdata-io atera.py
  ``_get_aligned_images``: "in the global (morphology px) coordinate system").
- mpp of the H&E: (global units per H&E pixel) x pixel_size, the global units per pixel being ``sqrt(|det A|)`` of the
  H&E's 2x2 affine onto ``global``, the same scale ``Rasterize`` uses (as in ``make_contract_fixture.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

ROLES = {"image": "^images/he_image$", "table": "^tables/table$"}
MPP_TARGET = 0.5


def chain(stain_json: Path | None = None):
    """default_chain(0.5) with squidpy's "weka" tissue detection instead of "otsu".

    On these slides otsu (squidpy detect_tissue's default) gives an empty mask for the brain (sddb_fmjq2bpd2q): its
    pale tissue is close to the glass, otsu keeps only nuclei (5.4% of the overview), and the area filter
    (min_specimen_area_frac 0.01) drops them all, so StainNormalize's fit fails ("[image_2] no tissue pixels for
    stain fitting"). weka gives 86% for the brain and follows the tissue outline on the other three (T144 task-7
    report, scratchpad diag of the overviews TissueMask sees).

    `stain_json` is a ``stain.json`` that an earlier run's ``fit_stain`` job wrote (copied out of the run state, which
    a rerun clears): StainNormalize then applies that pooled reference, and the rerun skips the fit.
    """
    from sddb.contract import AdapterRecord
    from sddb.loader import Centroids, Rasterize, StainNormalize, TissueMask, adapter_from_record

    stain = StainNormalize()
    if stain_json is not None:
        (record,) = json.loads(stain_json.read_text()).values()
        stain = adapter_from_record(AdapterRecord(**record))
    return [Rasterize(MPP_TARGET), Centroids(), TissueMask(method="weka"), stain]


class Member:
    """The part of :class:`sddb.Dataset` that ``materialize`` reads, as in tests/test_loader_materialize.py."""

    def __init__(self, uid: str, study_id: str, path: Path):
        self.uid, self.study_id, self._path = uid, study_id, path

    def to_sdata(self):
        """Open the store, with the table's X as a dask array.

        ``sd.read_zarr`` reads a table whole (anndata's ``read_zarr``), and ``materialize`` opens every member in
        the controller before it starts the jobs, so X is read lazily here (``anndata.experimental.read_elem_lazy``,
        CSR in chunks of 1000 rows): the controller then holds only obs and var. The jobs open the store themselves.
        """
        import anndata as ad
        import spatialdata as sd
        import zarr
        from anndata.experimental import read_elem_lazy
        from anndata.io import read_elem

        sdata = sd.read_zarr(self._path, selection=("images", "shapes"))
        g = zarr.open_group(self._path / "tables" / "table", mode="r")
        sdata.tables["table"] = ad.AnnData(
            X=read_elem_lazy(g["X"]), obs=read_elem(g["obs"]), var=read_elem(g["var"]), uns=read_elem(g["uns"])
        )
        return sdata


def store(bundle: Path, uid: str, stores: Path) -> None:
    """Read one Atera bundle and write its H&E, cell boundaries and table as ``<stores>/<uid>.zarr``."""
    import anndata as ad
    import spatialdata_io
    from scipy import sparse
    from spatialdata.transformations import get_transformation

    t0 = time.time()
    stores.mkdir(parents=True, exist_ok=True)
    spec = json.loads((bundle / "experiment.spatial").read_text())
    ps = float(spec["pixel_size"])
    sdata = spatialdata_io.atera(
        bundle,
        aligned_images=True,
        morphology_images=False,
        cells_labels=False,  # the table then annotates cell_boundaries (by cell_id): the cells come from the shapes
        nucleus_labels=False,
        nucleus_boundaries=False,
    )
    assert float(sdata.attrs["pixel_size"]) == ps, (sdata.attrs["pixel_size"], ps)
    table = sdata.tables["table"]
    # io atera: obs index name duplicates the barcode column; fixed upstream in the fork, remove this once the env pins
    # that commit (see make_contract_fixture.py).
    if table.obs.index.name in table.obs.columns:
        table.obs.index.name = None
    # The bundle's X has int64 indices: 12 bytes per nonzero with the int32 counts (1.48e9 nonzeros for the breast),
    # and anndata writes a dask sparse X with int64 indices whatever the blocks hold (anndata _io/specs/methods.py,
    # write_dask_sparse, as_int64_indices). So X is computed here, each block rebuilt with int32 indices (scipy's
    # constructor picks int32 when the values fit: _compressed.py, get_index_dtype with check_contents), and written
    # with anndata's write_csr_csc_indices_with_min_possible_dtype (uint16 for 24,677 genes). A job that reads it back
    # holds 8 bytes per nonzero (scipy keeps no uint16 indices).
    x = table.X.map_blocks(
        lambda b: sparse.csr_matrix((b.data, b.indices, b.indptr), shape=b.shape),
        dtype=table.X.dtype,
        meta=table.X._meta,
    ).compute()
    table.X = sparse.csr_matrix(x)
    del x
    he = sdata.images["he_image"]
    a = np.asarray(get_transformation(he, "global").to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
    he_scale = math.sqrt(abs(np.linalg.det(a[:2, :2])))  # global (morphology px) per H&E px
    info = {
        "uid": uid,
        "bundle": str(bundle),
        "experiment.spatial": {k: spec.get(k) for k in ("pixel_size", "region_name", "sample_type", "num_cells")},
        "table": list(table.shape),
        "x_type": type(table.X).__name__,
        "x_dtype": str(table.X.dtype),
        "x_indices_dtype": str(table.X.indices.dtype),
        "nnz": int(table.X.nnz),
        "he_scales": {k: list(v["image"].shape) for k, v in he.items()},
        "he_affine": a.tolist(),
        "he_global_per_px": he_scale,
        "mpp_native": he_scale * ps,
        "read_s": round(time.time() - t0, 1),
    }
    print(json.dumps(info, indent=1), flush=True)
    path = stores / f"{uid}.zarr"
    with ad.settings.override(write_csr_csc_indices_with_min_possible_dtype=True):
        sdata.subset(["he_image", "cell_boundaries", "table"]).write(path, overwrite=True)
    info["write_s"] = round(time.time() - t0 - info["read_s"], 1)
    info["store"] = str(path)
    info["store_bytes"] = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    (stores / f"{uid}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("write_s", "store", "store_bytes")}), flush=True)


def _members(stores: Path) -> tuple[list[Member], dict[str, float]]:
    infos = [json.loads(p.read_text()) for p in sorted(stores.glob("*.json"))]
    members = [Member(i["uid"], f"candidate:{i['uid']}", Path(i["store"])) for i in infos]
    return members, {i["uid"]: i["mpp_native"] for i in infos}


def run_materialize(stores: Path, dest: Path, workers: int, slurm: dict, stain_json: Path | None = None) -> None:
    """Materialize the stores as one collection with the H&E chain, on Slurm through snakemake."""
    from sddb.loader import materialize

    members, mpp = _members(stores)
    print(json.dumps({"members": [m.uid for m in members], "mpp_native": mpp, "slurm": slurm}), flush=True)
    t0 = time.time()
    materialize(
        members,
        dest,
        adapters=chain(stain_json),
        role="he",
        roles=ROLES,
        mpp_native=mpp,
        runner="snakemake",
        executor="slurm",
        workers=workers,
        slurm=slurm,
    )
    print(json.dumps({"materialize_s": round(time.time() - t0, 1)}), flush=True)


def check(stores: Path, dest: Path, out: Path) -> None:
    """Validate the collection, the cells in tissue per slide, StainNormalize's change and the tissue-mask figure."""
    import dask.array as da
    import zarr
    from anndata.io import read_elem

    from sddb.contract import validate
    from sddb.loader import LoaderItem, Rasterize

    out.mkdir(parents=True, exist_ok=True)
    members, mpp = _members(stores)
    summary: dict = {"validate": validate(dest)}
    print(json.dumps(summary), flush=True)
    root = zarr.open_group(dest, mode="r")
    slides = {}
    for key in sorted(k for k in root.group_keys() if k.startswith("dataset_")):
        g = root[key]
        contract = read_elem(g["uns"]["sddb_contract"])
        uid = contract["uid"]
        in_tissue = np.asarray(read_elem(g["obs"]["in_tissue"]))
        he = da.from_zarr(g["images"]["he"])
        tissue = da.from_zarr(g["labels"]["tissue"])
        slides[uid] = {
            "key": key,
            "cells": int(g["obsm"]["spatial"].shape[0]),
            "in_tissue_share": float(in_tissue.mean()),
            "he": list(he.shape),
            "tissue_px_share": float(tissue.mean().compute()),
        }
        # StainNormalize's change: the written image against Rasterize's output from the same source
        member = next(m for m in members if m.uid == uid)
        item = LoaderItem.from_sdata(
            member.to_sdata(), uid=uid, study_id=member.study_id, role="he", roles=ROLES, mpp_native=mpp[uid]
        )
        raw = Rasterize(MPP_TARGET).build(item).image.data
        assert raw.shape == he.shape, (raw.shape, he.shape)
        diff = da.abs(raw.astype(np.int16) - he.rechunk(raw.chunks).astype(np.int16))
        on = (tissue > 0).rechunk(raw.chunks[1:])[None]
        all_mean, tissue_sum, tissue_n = da.compute(diff.mean(), (diff * on).sum(), on.sum() * he.shape[0])
        slides[uid]["stain_mean_abs_change_all"] = float(all_mean)
        slides[uid]["stain_mean_abs_change_tissue"] = float(tissue_sum / max(int(tissue_n), 1))
        slides[uid]["stain_reference"] = contract.get("stain_reference")
        print(json.dumps({uid: slides[uid]}, default=str), flush=True)
    summary["slides"] = slides
    (out / "check.json").write_text(json.dumps(summary, indent=1, default=str))
    _tissue_figure(dest, slides, out / "tissue_masks.png")


def _tissue_figure(dest: Path, slides: dict, path: Path, side: int = 1200) -> None:
    """Each slide's H&E with its tissue mask outlined, downsampled to about `side` px, with spatialdata_plot."""
    import dask.array as da
    import matplotlib
    import zarr

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import spatialdata as sd
    import spatialdata_plot  # noqa: F401
    from spatialdata.models import Image2DModel, Labels2DModel

    root = zarr.open_group(dest, mode="r")
    fig, axs = plt.subplots(1, len(slides), figsize=(4 * len(slides), 4.5))
    for ax, (uid, s) in zip(np.atleast_1d(axs), slides.items(), strict=True):
        g = root[s["key"]]
        he = da.from_zarr(g["images"]["he"])
        f = max(1, math.ceil(max(he.shape[1:]) / side))
        small = np.asarray(he[:, ::f, ::f].compute())
        mask = np.asarray(da.from_zarr(g["labels"]["tissue"])[::f, ::f].compute())
        view = sd.SpatialData(
            images={"he": Image2DModel.parse(small, dims=("c", "y", "x"))},
            labels={"tissue": Labels2DModel.parse(mask, dims=("y", "x"))},
        )
        (
            view.pl.render_images("he")
            .pl.render_labels("tissue", outline=True, outline_color="lime", fill_alpha=0.0, contour_px=2)
            .pl.show(ax=ax, title="", colorbar=False)
        )
        ax.set_title(f"{uid}\nin tissue {s['in_tissue_share']:.1%} of {s['cells']:,} cells", fontsize=9)
        ax.set_axis_off()
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse the command line and run one step."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("store")
    s.add_argument("bundle", type=Path)
    s.add_argument("uid")
    s.add_argument("stores", type=Path)
    m = sub.add_parser("materialize")
    m.add_argument("stores", type=Path)
    m.add_argument("dest", type=Path)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--slurm-json", required=True, help="the slurm resources of materialize, as JSON")
    m.add_argument("--stain-json", type=Path, help="apply the reference in this stain.json instead of fitting one")
    c = sub.add_parser("check")
    c.add_argument("stores", type=Path)
    c.add_argument("dest", type=Path)
    c.add_argument("out", type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.step == "store":
        store(args.bundle, args.uid, args.stores)
    elif args.step == "materialize":
        run_materialize(args.stores, args.dest, args.workers, json.loads(args.slurm_json), args.stain_json)
    else:
        check(args.stores, args.dest, args.out)


if __name__ == "__main__":
    main()
