"""Materialize the published Atera H&E slides into a loader collection on Slurm, and check the result.

Two steps, each run on a compute node (see ``docs/use_cases/loader_atera.md``)::

    python scripts/atera_loader_cohort.py materialize <dest> [--version 2026-11-12] [--workers 1] --slurm-json '{...}'
    python scripts/atera_loader_cohort.py check <dest> <out dir> [--version 2026-11-12]

``materialize`` takes the 10x Atera records of a catalog release (``technology == "10x_atera"``) as a cohort and runs
``SpatialDataCohort.materialize`` with ``default_chain(0.5)``, the snakemake runner and the Slurm executor. Each
member's store is copied from the release into the sddb cache (``SDDB_CACHE_DIR``) once and opened from there
(``Dataset._local_store``, ``sddb/cohort.py`` ``_cached_store``). ``check`` validates the collection and compares
every dataset with its catalog record: the cells, the features, the filled features, the centroid fallbacks, the
tissue share, and the change made by stain normalisation; it also draws the tissue masks.

Sources (read 2026-11-13):

- microns per H&E pixel: the catalog's ``mpp`` column. For the four Atera v1 records it equals the value derived in
  v1 from the bundle (``experiment.spatial`` ``pixel_size`` times ``sqrt(|det A|)`` of the H&E affine onto
  ``global``, see the git history of this script) to 1e-15.
- roles: the six stores have ``images/he_image``, ``labels/cell_labels`` and ``tables/table`` (catalog
  ``sdata_repr``). The Atera v1 tables annotate ``cell_labels`` (instance key ``label_id``); the two WTA Preview
  tables, read with spatialdata-io's xenium reader, annotate ``cell_circles`` by ``cell_id`` and hold the label
  value in ``obs["cell_labels"]`` (``tables/table/zarr.json`` of each store), which ``LoaderItem.from_sdata`` reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

ROLES = {"image": "^images/he_image$", "table": "^tables/table$", "labels": "^labels/cell_labels$"}
MPP_TARGET = 0.5
TECHNOLOGY = "10x_atera"
VERSION = "2026-11-12"


def cohort(version: str):
    """The Atera records of catalog release `version`, and each one's microns per H&E pixel (catalog ``mpp``)."""
    import sddb

    catalog = sddb.Catalog(version=version)
    members = catalog.query(technology=TECHNOLOGY)
    df = members.to_df()
    mpp = {str(u): float(m) for u, m in zip(df["uid"], df["mpp"], strict=True)}
    return catalog, members, mpp


def run_materialize(dest: Path, version: str, workers: int, slurm: dict) -> None:
    """Materialize the Atera records of the release with the default H&E chain, on Slurm through snakemake."""
    from sddb.loader import default_chain

    catalog, members, mpp = cohort(version)
    df = members.to_df()
    print(
        json.dumps(
            {
                "catalog": catalog.url,
                "release": catalog.release,
                "members": df[["uid", "title", "n_obs", "n_features", "mpp", "size_bytes"]].to_dict("records"),
                "slurm": slurm,
            },
            default=str,
        ),
        flush=True,
    )
    t0 = time.time()
    members.materialize(
        dest,
        adapters=default_chain(MPP_TARGET),
        role="he",
        roles=ROLES,
        mpp_native=mpp,
        runner="snakemake",
        executor="slurm",
        workers=workers,
        slurm=slurm,
    )
    print(json.dumps({"materialize_s": round(time.time() - t0, 1)}), flush=True)


def check(dest: Path, out: Path, version: str) -> None:
    """Validate the collection and compare each dataset with its catalog record; draw the tissue masks."""
    import dask.array as da
    import zarr
    from anndata.io import read_elem

    from sddb.contract import validate
    from sddb.loader import LoaderItem, Rasterize
    from sddb.loader._materialize import _read_store

    out.mkdir(parents=True, exist_ok=True)
    _, members, mpp = cohort(version)
    records = {str(m.uid): m for m in members}
    summary: dict = {"validate": validate(dest)}
    print(json.dumps(summary), flush=True)
    root = zarr.open_group(dest, mode="r")
    keys = sorted((k for k in root.group_keys() if k.startswith("dataset_")), key=lambda k: int(k.split("_")[1]))
    slides = {}
    for key in keys:
        g = root[key]
        contract = read_elem(g["uns"]["sddb_contract"])
        uid = contract["uid"]
        record = records[uid]
        var = read_elem(g["var"])
        present = var["present"].to_numpy(dtype=bool)
        in_tissue = np.asarray(read_elem(g["obs"]["in_tissue"]))
        he = da.from_zarr(g["images"]["he"])
        tissue = da.from_zarr(g["labels"]["tissue"])
        s = {
            "key": key,
            "title": record.title,
            "cells_written": int(g["obsm"]["spatial"].shape[0]),
            "cells_catalog": int(record.n_obs),
            "features_written": int(len(var)),
            "features_catalog": int(record.n_features),
            "present": int(present.sum()),
            "filled": int((~present).sum()),
            "n_features_present": contract.get("n_features_present"),
            "n_features_filled": contract.get("n_features_filled"),
            "n_centroid_fallback": contract.get("n_centroid_fallback"),
            "n_outside_image": contract.get("n_outside_image"),
            "outside_image_flags": int(np.asarray(read_elem(g["obs"]["outside_image"])).sum()),
            "in_tissue_share": float(in_tissue.mean()),
            "he": list(he.shape),
            "tissue_px_share": float(tissue.mean().compute()),
            "stain_reference": contract.get("stain_reference"),
        }
        s["cells_equal"] = s["cells_written"] == s["cells_catalog"]
        s["present_equal_catalog_features"] = s["present"] == s["features_catalog"]
        s["filled_equal_contract"] = s["filled"] == s["n_features_filled"]
        # StainNormalize's change: the written image against Rasterize's output from the same source store
        item = LoaderItem.from_sdata(
            _read_store(record._local_store()),
            uid=uid,
            study_id=str(record.study_id),
            role="he",
            roles=ROLES,
            mpp_native=mpp[uid],
        )
        raw = Rasterize(MPP_TARGET).build(item).image.data
        assert raw.shape == he.shape, (raw.shape, he.shape)
        diff = da.abs(raw.astype(np.int16) - he.rechunk(raw.chunks).astype(np.int16))
        on = (tissue > 0).rechunk(raw.chunks[1:])[None]
        tissue_sum, tissue_n = da.compute((diff * on).sum(), on.sum() * he.shape[0])
        s["stain_mean_abs_change_tissue"] = float(tissue_sum / max(int(tissue_n), 1))
        slides[uid] = s
        print(json.dumps({uid: s}, default=str), flush=True)
    summary["union"] = len(read_elem(root[keys[0]]["var"]))
    summary["slides"] = slides
    (out / "check.json").write_text(json.dumps(summary, indent=1, default=str))
    _tissue_figure(dest, slides, out / "tissue_masks.png")


def _tissue_figure(dest: Path, slides: dict, path: Path, side: int = 900) -> None:
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
    cols = 2 if len(slides) == 4 else min(3, len(slides))
    rows = math.ceil(len(slides) / cols)
    hw = [root[s["key"]]["images"]["he"].shape[1:] for s in slides.values()]
    aspect = max(h / w for h, w in hw)  # height of a panel per unit of width
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, (4 * aspect + 0.6) * rows), squeeze=False)
    for ax in np.ravel(axs)[len(slides) :]:
        ax.set_axis_off()
    for ax, s in zip(np.ravel(axs), slides.values(), strict=False):
        g = root[s["key"]]
        he = da.from_zarr(g["images"]["he"])
        f = max(1, math.ceil(max(he.shape[1:]) / side))
        small = np.asarray(he[:, ::f, ::f].compute())
        mask = np.asarray(da.from_zarr(g["labels"]["tissue"])[::f, ::f].compute())
        view = sd.SpatialData(
            images={"he": Image2DModel.parse(small, dims=("c", "y", "x"), c_coords=["r", "g", "b"])},
            labels={"tissue": Labels2DModel.parse(mask, dims=("y", "x"))},
        )
        (
            view.pl.render_images("he")
            .pl.render_labels("tissue", outline=True, outline_color="lime", fill_alpha=0.0, contour_px=2)
            .pl.show(ax=ax, title="", colorbar=False)
        )
        title = s["title"].replace("Atera In Situ Gene Expression, ", "").replace("Preview Data: ", "Preview: ")
        ax.set_title(f"{title}\ntissue {s['tissue_px_share']:.0%} of the image", fontsize=9)
        ax.set_axis_off()
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse the command line and run one step."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("dest", type=Path)
    m.add_argument("--version", default=VERSION, help="the catalog release")
    m.add_argument("--workers", type=int, default=1)
    m.add_argument("--slurm-json", required=True, help="the slurm resources of materialize, as JSON")
    c = sub.add_parser("check")
    c.add_argument("dest", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--version", default=VERSION, help="the catalog release")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.step == "materialize":
        run_materialize(args.dest, args.version, args.workers, json.loads(args.slurm_json))
    else:
        check(args.dest, args.out, args.version)


if __name__ == "__main__":
    main()
