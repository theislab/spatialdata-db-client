"""Synthetic catalog/gene-index fixtures (test-only, not shipped)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sddb._catalog_schema import CATALOG_COLUMNS, COLLECTION_COLUMNS, FEATURE_INDEX_COLUMNS, cast_scalars

# study_id per fixture row (first len(_ROWS) used; pairs share a study for the citation tests)
_STUDIES = ("Smith2023", "Smith2023", "Lee2022", "Lee2022", "Wong2021")

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
            "license_unknown": [r[7] is None for r in _ROWS],
            "total_counts": [float(r[8]) * 100 for r in _ROWS],
            "study_id": [_STUDIES[i] if i < len(_STUDIES) else f"Study{i}" for i in range(n)],
            "vitessce_url": [f"s3://scverse-spatial-eu-central-1/.lamindb/{r[0]}_vitessce.json" for r in _ROWS],
            "flag_foo": [True] * n,  # unknown extra column
            # imaging metadata (typed nulls: not populated by the fixture)
            "mpp": [None] * n,
            "image_width": [None] * n,
            "image_height": [None] * n,
            "magnification": [None] * n,
            "sdata_repr": [None] * n,
            "n_transcripts": [None] * n,
            "capture_area": [None] * n,
            # study + per-sample condition columns (optional; null in the base fixture)
            **{
                c: [None] * n
                for c in ("sample_title", "timepoint", "timepoint_kind", "treatment", "genotype", "condition")
            },
            "timepoint_days": [None] * n,
            "obs_unit": ["cells"] * n,
            "segmentation_source": ["vendor"] * n,
            "segmentation_method": [None] * n,
            # superset list columns, derived per-row so the fixture stays len(_ROWS)-agnostic
            "collections": [[f"c-{i}"] if i % 3 else [] for i in range(n)],
            "staining_method": [["H&E", "IF"] if i % 2 else None for i in range(n)],
        }
    )
    return cast_scalars(df, CATALOG_COLUMNS)


def make_fixture_collections() -> pd.DataFrame:
    """Build a valid synthetic collections table (one row per collection)."""
    rows = [
        (
            "c-smith",
            "Smith2023",
            "Smith 2023 atlas",
            "publication",
            "EpCAM atlas",
            "10.1/smith",
            "DOI",
            "Xenium",
            "human",
            "lung",
            2,
            2,
            180_000,
            ["uid0001", "uid0002"],
        ),  # abstract/study_*/study-condition columns are optional: null here
        (
            "c-lee",
            "Lee2022",
            "Lee 2022 brain",
            "replicates",
            None,
            None,
            None,
            "Visium",
            "mouse",
            "brain",
            1,
            1,
            4_000,
            ["uid0003"],
        ),
    ]
    df = pd.DataFrame(rows, columns=list(COLLECTION_COLUMNS)[: len(rows[0])]).reindex(columns=list(COLLECTION_COLUMNS))
    for c, t in COLLECTION_COLUMNS.items():
        if t.startswith("list"):
            df[c] = df[c].astype(object)  # all-null reindexed columns default to float64
    return cast_scalars(df, COLLECTION_COLUMNS)


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


def make_fixture_feature_index() -> pd.DataFrame:
    """Small multi-modal feature index keyed to the fixture catalog uids."""
    rows = [
        # raw_label, type, modality, uid, counts, frac, ref_ids, ref_id_types, ref_relations, ref_conf, gene_xref
        (
            "EPCAM",
            "gene",
            "rna",
            "uid0001",
            5000.0,
            0.4,
            ["ENSG00000119888"],
            ["ensembl_gene"],
            ["is"],
            [1.0],
            ["ENSG00000119888"],
        ),
        (
            "EPCAM",
            "gene",
            "rna",
            "uid0002",
            3000.0,
            0.3,
            ["ENSG00000119888"],
            ["ensembl_gene"],
            ["is"],
            [1.0],
            ["ENSG00000119888"],
        ),
        (
            "anti-EpCAM",
            "protein",
            "protein",
            "uid0003",
            90.0,
            0.8,
            ["P16422"],
            ["uniprot"],
            ["targets"],
            [0.9],
            ["ENSG00000119888"],
        ),
        (
            "EPCAM",
            "gene_activity",
            "atac",
            "uid0005",
            10.0,
            0.1,
            ["ENSG00000119888"],
            ["ensembl_gene"],
            ["is"],
            [1.0],
            ["ENSG00000119888"],
        ),
        (
            "KRT8",
            "gene",
            "rna",
            "uid0001",
            9000.0,
            0.6,
            ["ENSG00000170421"],
            ["ensembl_gene"],
            ["is"],
            [1.0],
            ["ENSG00000170421"],
        ),
        ("Glucose", "metabolite", "msi", "uid0004", 500.0, 0.9, ["HMDB0000122"], ["hmdb"], ["is"], [1.0], []),
    ]
    cols = list(FEATURE_INDEX_COLUMNS)
    df = pd.DataFrame([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]) for r in rows], columns=cols)
    for c in ("raw_label", "feature_type", "modality", "uid"):
        df[c] = df[c].astype("string")
    for c in ("total_counts", "fraction_obs_detected"):
        df[c] = df[c].astype("Float64")
    return df


def write_fixture_feature_index(path: str | Path) -> Path:
    """Write the fixture feature index to parquet and return its path."""
    p = Path(path)
    make_fixture_feature_index().to_parquet(p)
    return p
