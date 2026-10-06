"""Synthetic catalog/gene-index fixtures (test-only, not shipped)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sddb._catalog_schema import CATALOG_COLUMNS, COLLECTION_COLUMNS

_URL = "s3://scverse-spatial-eu-central-1/.lamindb/{}.zarr"

# license id -> (url, noncommercial, redistributable); the producer derives these from the engine's
# licenses vocab, but the client-only fixture hardcodes the few ids it uses.
_LIC_PROPS: dict[str, tuple[str, bool, bool]] = {
    "CC-BY-4.0": ("https://creativecommons.org/licenses/by/4.0/", False, True),
    "CC-BY-NC-SA-4.0": ("https://creativecommons.org/licenses/by-nc-sa/4.0/", True, True),
    "CC0-1.0": ("https://creativecommons.org/publicdomain/zero/1.0/", False, True),
    "unknown": ("", False, False),
}

_ROWS = [
    ("uid0001", "human", "lung", "Xenium", "10x Xenium", "pass", "silver", "CC-BY-4.0", 100_000, 5000),
    ("uid0002", "human", "breast", "Xenium", "10x Xenium", "pass", "gold", "CC-BY-NC-SA-4.0", 80_000, 5000),
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
            "license": [(r[7] or "unknown") for r in _ROWS],
            "license_url": [_LIC_PROPS[r[7] or "unknown"][0] for r in _ROWS],
            "license_noncommercial": [_LIC_PROPS[r[7] or "unknown"][1] for r in _ROWS],
            "license_redistributable": [_LIC_PROPS[r[7] or "unknown"][2] for r in _ROWS],
            "n_obs": [r[8] for r in _ROWS],
            "n_features": [r[9] for r in _ROWS],
            "zarr_url": [_URL.format(r[0]) for r in _ROWS],
            "license_unknown": [False, False, False, True, False],
            "total_counts": [float(r[8]) * 100 for r in _ROWS],
            "study_id": ["Smith2023", "Smith2023", "Lee2022", "Lee2022", "Wong2021"],
            "vitessce_url": [f"s3://scverse-spatial-eu-central-1/.lamindb/{r[0]}_vitessce.json" for r in _ROWS],
            "flag_foo": [True] * n,  # unknown extra column
            # superset list columns (exercise list[string] validation)
            "collections": [["c-smith"], ["c-smith"], ["c-lee"], [], ["c-wong"]],
            "staining_method": [["H&E"], ["H&E", "IF"], None, ["IF"], ["H&E"]],
        }
    )
    # list[string] is not a pandas dtype — only cast the scalar columns.
    known = {c: t for c, t in CATALOG_COLUMNS.items() if c in df.columns and not t.startswith("list")}
    return df.astype(known)


def make_fixture_collections() -> pd.DataFrame:
    """Build a valid synthetic collections table (one row per collection)."""
    rows = [
        ("c-smith", "Smith2023", "Smith 2023 atlas", "publication", "EpCAM atlas",
         "10.1/smith", "DOI", "Xenium", "human", "lung", 2, 2, 180_000, ["uid0001", "uid0002"]),
        ("c-lee", "Lee2022", "Lee 2022 brain", "replicates", None, None, None,
         "Visium", "mouse", "brain", 1, 1, 4_000, ["uid0003"]),
    ]
    df = pd.DataFrame(rows, columns=list(COLLECTION_COLUMNS))
    scalar = {c: t for c, t in COLLECTION_COLUMNS.items() if not t.startswith("list")}
    return df.astype(scalar)


def write_fixture_collections(path: str | Path) -> Path:
    """Write the fixture collections table to parquet and return its path."""
    p = Path(path)
    make_fixture_collections().to_parquet(p)
    return p


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


def write_fixture_gene_index(path: str | Path) -> Path:
    """Write the fixture gene index to parquet and return its path."""
    p = Path(path)
    make_fixture_gene_index().to_parquet(p)
    return p


def make_fixture_citations_bib() -> str:
    """BibTeX keyed by the fixture study_ids, with nested braces and one extra study."""
    return (
        "@comment{ignored}\n\n"
        "@article{Smith2023,\n  title = {Spatial {EpCAM} atlas},\n  author = {Smith, A. and {The Consortium}},\n"
        "  year = {2023}\n}\n\n"
        "@article{Lee2022,\n  title = {Brain {Visium} {{nested}}},\n  year = {2022}\n}\n\n"
        "@article{Wong2021,\n  title = {Colon},\n  year = {2021}\n}\n\n"
        "@article{Extra1999,\n  title = {Not in cohort},\n  year = {1999}\n}\n"
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
