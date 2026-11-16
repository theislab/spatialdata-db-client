"""Offline fixtures: a small catalog parquet and local SpatialData stores, built as the client's tests do."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData

from sddb._catalog_schema import CATALOG_COLUMNS, cast_scalars

# licence id -> (noncommercial, redistributable, unknown)
LICENSES = {
    "CC-BY-4.0": (False, True, False),
    "CC-BY-NC-4.0": (True, True, False),
    "CC0-1.0": (False, True, False),
    "unknown": (False, False, True),
}
SPOT_GENES = [f"ENSG{i:011d}" for i in range(1, 9)]
IMAGING_GENES = SPOT_GENES[:2] + [f"ENSG{i:011d}" for i in range(100, 104)]


def symbols(genes):
    """The fixture gene symbol of each Ensembl id."""
    return [f"G{g[-3:]}" for g in genes]


def write_store(path, genes, n, *, ensembl=True, keys=True):
    """A store with one table and stored ``obsm["spatial"]``.

    By default var_names are Ensembl ids with symbols in ``var["symbol"]`` (HEST style). ``ensembl=False``
    puts the symbols in var_names with no Ensembl column (Allen MERSCOPE style). ``keys=False`` leaves
    var_names as Ensembl ids with no symbol column, so the table has no gene symbols.
    """
    var = pd.DataFrame({"symbol": symbols(genes)}, index=genes) if keys else pd.DataFrame(index=genes)
    if not ensembl:
        var = pd.DataFrame(index=symbols(genes))
    table = ad.AnnData(
        X=np.ones((n, len(genes)), "float32"),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=var,
        obsm={"spatial": np.arange(2 * n, dtype="float64").reshape(n, 2)},
    )
    SpatialData(tables={"table": table}).write(path)
    return str(path)


def write_spatial_store(path, genes, X, *, ensembl=True, lowres=False):
    """A store with a ``table`` that annotates a ``cells`` circles element, plus an optional low-res image.

    ``genes`` are symbols. With ``ensembl=True`` var_names are Ensembl ids with the symbols in
    ``var["symbol"]`` (HEST style); otherwise var_names are the symbols.
    """
    from spatialdata.models import Image2DModel, ShapesModel, TableModel

    n = X.shape[0]
    ids = [f"ENSG{i:011d}" for i in range(1, len(genes) + 1)]
    var = pd.DataFrame({"symbol": genes}, index=ids) if ensembl else pd.DataFrame(index=genes)
    xy = np.random.default_rng(n).uniform(0, 100, (n, 2))
    obs = pd.DataFrame(
        {"region": pd.Categorical(["cells"] * n), "cell_id": np.arange(n)}, index=[f"c{i}" for i in range(n)]
    )
    table = TableModel.parse(
        ad.AnnData(X=np.asarray(X, "float32"), obs=obs, var=var, obsm={"spatial": xy}),
        region="cells",
        region_key="region",
        instance_key="cell_id",
    )
    shapes = ShapesModel.parse(xy, geometry=0, radius=np.full(n, 2.0), index=np.arange(n))
    images = {}
    if lowres:
        images["tissue_lowres_image"] = Image2DModel.parse(
            np.random.default_rng(0).integers(0, 255, (3, 50, 50), dtype=np.uint8), dims=("c", "y", "x")
        )
    SpatialData(tables={"table": table}, shapes={"cells": shapes}, images=images).write(path)
    return str(path)


def member(
    uid,
    technology="Visium",
    *,
    study_id=None,
    collections=None,
    doi=None,
    license="CC-BY-4.0",
    n=4,
    ensembl=True,
    keys=True,
    organism="human",
):
    """One fixture member; ``study_id`` defaults to a study of its own. See ``write_store`` for the flags."""
    return {
        "ensembl": ensembl,
        "keys": keys,
        "organism": organism,
        "uid": uid,
        "technology": technology,
        "study_id": f"geo:{uid}" if study_id is None else study_id or None,
        "collections": collections or [],
        "doi": doi,
        "license": license,
        "n": n,
    }


@pytest.fixture
def make_catalog(tmp_path, tmp_path_factory, monkeypatch):
    """Build ``sddb.Catalog`` from member dicts; uids in ``broken`` point at a store without a table."""
    import sddb

    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path_factory.mktemp("sddb-cache")))

    builds = iter(range(1000))

    def build(members, broken=()):
        root = tmp_path / f"build{next(builds)}"
        root.mkdir()
        rows = []
        for m in members:
            genes = SPOT_GENES if m["technology"] in ("Visium", "VisiumHD", "SpatialTranscriptomics") else IMAGING_GENES
            path = root / f"{m['uid']}.zarr"
            if "store" in m:
                genes, url = m["genes"], m["store"](path)
            elif m["uid"] in broken:
                path.mkdir()
                url = str(path)
            else:
                url = write_store(path, genes, m["n"], ensembl=m["ensembl"], keys=m["keys"])
            nc, redist, unknown = LICENSES[m["license"]]
            rows.append(
                {
                    "uid": m["uid"],
                    "title": f"Section {m['uid']}",
                    "organism": m["organism"],
                    "tissue": m.get("tissue", "colon"),
                    "technology": m["technology"],
                    "assay": m["technology"],
                    "validation_status": "pass",
                    "license": m["license"],
                    "license_noncommercial": nc,
                    "license_redistributable": redist,
                    "license_unknown": unknown,
                    "n_obs": m["n"],
                    "n_features": len(genes),
                    "zarr_url": url,
                    "study_id": m["study_id"],
                    "doi": m["doi"],
                    "donor_id": f"donor-{m['uid']}",
                    "collections": list(m["collections"]),
                }
            )
        df = cast_scalars(pd.DataFrame(rows), CATALOG_COLUMNS)
        p = root / "catalog.parquet"
        df.to_parquet(p)
        return sddb.Catalog.from_file(p)

    return build
