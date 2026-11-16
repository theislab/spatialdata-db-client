"""Every documented query runs on a schema-3 catalog (vendor-prefixed technology ids, catalog-2026-11-12 on)."""

import re
from pathlib import Path

import pandas as pd
import pytest

from sddb import Catalog
from sddb._catalog_schema import CATALOG_COLUMNS, RANGE_COLUMNS
from sddb.catalog import resolve_facet_values

from ._fixtures import make_fixture_catalog

_ROOT = Path(__file__).parent.parent

# (technology, assay) pairs of catalog-2026-11-12 (schema 3), read from its catalog.parquet
_PAIRS_2026_11_12 = [
    ("10x_atera", "10x Atera"),
    ("10x_spatial_transcriptomics", "spatial transcriptomics"),
    ("10x_visium", "Visium Spatial Gene Expression"),
    ("10x_visium_hd", "Visium HD"),
    ("10x_xenium", "10x Xenium"),
    ("akoya_phenocycler", "PhenoCycler"),
    ("akoya_phenocycler", "PhenoCycler-Fusion"),
    ("bruker_cosmx", "CosMx"),
    ("dbit_seq", "DBiT-seq"),
    ("ionpath_mibiscope", "MIBI-TOF"),
    ("maldi_msi", "MALDI mass spectrometry imaging"),
    ("merfish", "MERFISH"),
    ("sbt_hyperion", "Imaging Mass Cytometry"),
    ("slide_seq", "Slide-seqV2"),
    ("spatial_atac_seq", "spatial ATAC-seq"),
    ("stomics_stereo_seq", "Stereo-seq"),
    ("vizgen_merscope", "Vizgen MERSCOPE"),
]
_TECH = pd.DataFrame(_PAIRS_2026_11_12, columns=["technology", "assay"])


def _resolve(value: str) -> list[str]:
    return resolve_facet_values(_TECH, "technology", [value])


@pytest.mark.parametrize(
    ("value", "ids"),
    [
        ("Visium", ["10x_visium"]),  # schema-2 token = id without its vendor prefix
        ("VisiumHD", ["10x_visium_hd"]),
        ("Merscope", ["vizgen_merscope"]),
        ("Xenium", ["10x_xenium"]),
        ("SpatialTranscriptomics", ["10x_spatial_transcriptomics"]),
        ("10x Xenium", ["10x_xenium"]),  # assay label
        ("vizgen_merscope", ["vizgen_merscope"]),  # the id itself
        ("MaldiMsi", ["maldi_msi"]),  # no vendor prefix: full id only
    ],
)
def test_schema2_tokens_and_labels_resolve_to_schema3_ids(value, ids):
    assert _resolve(value) == ids


def test_unprefixed_ids_get_no_partial_alias():
    with pytest.raises(ValueError, match="unknown technology value 'seq'"):
        _resolve("seq")  # dbit_seq / slide_seq are not vendor-prefixed


def test_every_technology_literal_in_the_docs_resolves():
    files = [_ROOT / "README.md", *(_ROOT / "docs").rglob("*.md")]
    found = {
        v
        for f in files
        if "_build" not in f.parts and "superpowers" not in f.parts
        for v in re.findall(r'technology="([^"]+)"', f.read_text())
    }
    assert found, "no documented technology= query found"
    for v in sorted(found):
        assert _resolve(v), v


@pytest.fixture
def schema3_catalog(tmp_path) -> Catalog:
    df = make_fixture_catalog()
    ids = {"Visium": "10x_visium", "Xenium": "10x_xenium"}
    assert set(df["technology"]) <= set(ids), set(df["technology"])
    df["technology"] = df["technology"].map(ids)
    df.loc[df.index[0], ["technology", "assay", "tissue"]] = ["10x_visium", "Visium Spatial Gene Expression", "lung"]
    df.loc[df.index[-1], ["technology", "assay"]] = ["vizgen_merscope", "Vizgen MERSCOPE"]  # row 1 stays Xenium
    df["n_cells"] = pd.array([150_000] + [None] * (len(df) - 1), dtype="Int64")
    p = tmp_path / "catalog.parquet"
    df.to_parquet(p)
    return Catalog.from_file(p)


@pytest.mark.parametrize(
    "facets",
    [
        {"technology": "Visium", "tissue": "lung"},  # docs/index.md, docs/intro/db-stack.md
        {"technology": "Visium"},  # docs/intro/db-stack.md
        {"technology": "vizgen_merscope"},  # docs/start.md
        {"technology": "Merscope"},  # docs/start.md
        {"technology": "Xenium"},
    ],
)
def test_documented_queries_run_on_a_schema3_catalog(schema3_catalog, facets):
    assert len(schema3_catalog.query(**facets)) >= 1


def test_release_metric_columns_are_known_and_rangeable(schema3_catalog):
    for col in (
        "n_cells",
        "median_transcripts_per_cell",
        "median_genes_per_cell",
        "decoded_transcripts",
        "imaged_area",
        "onboard_version",
        "panel",
        "review_reasons",
        "review_classes",
    ):
        assert col in CATALOG_COLUMNS, col
    assert "n_cells" in RANGE_COLUMNS
    assert len(schema3_catalog.query(n_cells__gte=100_000)) == 1


def test_dates_section_says_publication_date_is_the_release_date():
    text = (_ROOT / "docs" / "reference" / "catalog-columns.md").read_text()
    assert "## Dates" in text and "source release date" in text and "`publication_year`" in text
