"""Synthetic catalog/gene-index fixtures (test-only, not shipped)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sddb._catalog_schema import CATALOG_COLUMNS

_URL = "s3://scverse-spatial-eu-central-1/.lamindb/{}.zarr"

_ROWS = [
    ("uid0001", "human", "lung", "Xenium", "10x Xenium", "pass", "silver", "CC-BY-4.0", 100_000, 5000),
    ("uid0002", "human", "breast", "Xenium", "10x Xenium", "pass", "gold", "CC-BY-4.0", 80_000, 5000),
    (
        "uid0003",
        "mouse",
        "brain",
        "Visium",
        "Visium Spatial Gene Expression",
        "pass",
        "bronze",
        "CC0-1.0",
        4_000,
        32000,
    ),
    ("uid0004", "mouse", "liver", "Visium", "Visium Spatial Gene Expression", "fail", None, None, 3_500, 32000),
    (
        "uid0005",
        "human",
        "colon",
        "Visium",
        "Visium Spatial Gene Expression",
        "pass",
        "silver",
        "CC-BY-4.0",
        4_200,
        18000,
    ),
]


def make_fixture_catalog() -> pd.DataFrame:
    """Build a valid synthetic catalog with extras and optional columns."""
    n = len(_ROWS)
    df = pd.DataFrame(
        {
            "uid": [r[0] for r in _ROWS],
            "organism": [r[1] for r in _ROWS],
            "tissue": [r[2] for r in _ROWS],
            "technology": [r[3] for r in _ROWS],
            "assay": [r[4] for r in _ROWS],
            "validation_status": [r[5] for r in _ROWS],
            "tier": [r[6] for r in _ROWS],
            "license_spdx": [r[7] for r in _ROWS],
            "n_obs": [r[8] for r in _ROWS],
            "n_features": [r[9] for r in _ROWS],
            "zarr_url": [_URL.format(r[0]) for r in _ROWS],
            "license_unknown": [False, False, False, True, False],
            "total_counts": [float(r[8]) * 100 for r in _ROWS],
            "flag_foo": [True] * n,  # unknown extra column
        }
    )
    known = {c: t for c, t in CATALOG_COLUMNS.items() if c in df.columns}
    return df.astype(known)


def make_fixture_gene_index() -> pd.DataFrame:
    """Build a synthetic gene index keyed to the fixture catalog uids."""
    rows = [
        ("EPCAM", "ENSG00000119888", "uid0001", 5000.0, 0.4),
        ("EPCAM", "ENSG00000119888", "uid0002", 3000.0, 0.3),
        ("KRT8", "ENSG00000170421", "uid0001", 9000.0, 0.6),
        ("PTPRC", "ENSG00000081237", "uid0002", 1200.0, 0.2),
        ("Snap25", "ENSMUSG00000027273", "uid0003", 800.0, 0.5),
        ("Gfap", "ENSMUSG00000020932", "uid0003", 600.0, 0.3),
        ("Alb", "ENSMUSG00000029368", "uid0004", 2000.0, 0.7),
        ("EPCAM", "ENSG00000119888", "uid0005", 700.0, 0.25),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "feature_id", "uid", "total_counts", "fraction_obs_detected"])
    return df.astype(
        {
            "symbol": "string",
            "feature_id": "string",
            "uid": "string",
            "total_counts": "Float64",
            "fraction_obs_detected": "Float64",
        }
    )


def write_fixture_catalog(path: str | Path) -> Path:
    """Write the fixture catalog to parquet and return its path."""
    p = Path(path)
    make_fixture_catalog().to_parquet(p)
    return p


def make_tiny_sdata_zarr(tmp_path: Path) -> Path:
    """Write a tiny SpatialData (3x8x8 image + 4x3 table) to ``tmp_path/tiny.zarr``."""
    import anndata as ad
    import numpy as np
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel

    image = Image2DModel.parse(np.zeros((3, 8, 8), dtype="uint8"), dims=("c", "y", "x"))
    table = ad.AnnData(X=np.ones((4, 3), dtype="float32"))
    sdata = SpatialData(images={"img": image}, tables={"table": table})
    path = tmp_path / "tiny.zarr"
    sdata.write(path)
    return path
