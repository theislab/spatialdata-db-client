import pandas as pd
import pytest

from sddb import _catalog_schema as schema
from sddb.catalog import Catalog


def test_study_id_and_doi_are_facets():
    assert "study_id" in schema.FACETS
    assert "doi" in schema.FACETS


def test_query_by_study_id(tmp_path):
    df = pd.DataFrame(
        {
            "uid": ["u1", "u2"],
            "study_id": ["oliveira_2025", "other_2024"],
            "doi": ["10.1/x", "10.2/y"],
            "technology": ["Visium HD"] * 2,
            "assay": ["spatial"] * 2,
            "organism": ["human"] * 2,
            "validation_status": ["pass"] * 2,
            "zarr_url": ["s3://b/1.zarr", "s3://b/2.zarr"],
        }
    )
    p = tmp_path / "catalog.parquet"
    df.to_parquet(p)
    cat = Catalog.from_file(p)
    assert list(cat.query(study_id="oliveira_2025").to_df()["uid"]) == ["u1"]
    assert list(cat.query(doi="10.2/y").to_df()["uid"]) == ["u2"]


def test_query_by_tissue_and_disease_rollup(tmp_path):
    df = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u3"],
            "technology": ["Xenium"] * 3,
            "assay": ["spatial"] * 3,
            "organism": ["Mus musculus"] * 3,
            "tissue": ["cerebellum", "hippocampal formation", "liver"],
            "tissue_system": ["nervous system", "nervous system", "digestive system"],
            "tissue_organ": ["brain", "brain", "liver"],
            "disease_class": ["neurodegenerative disease", None, None],
            "disease_group": ["Alzheimer disease", None, None],
            "validation_status": ["pass"] * 3,
            "zarr_url": ["s3://b/1.zarr", "s3://b/2.zarr", "s3://b/3.zarr"],
        }
    )
    p = tmp_path / "catalog.parquet"
    df.to_parquet(p)
    cat = Catalog.from_file(p)
    # the form the MCP connector emits
    assert list(cat.query(tissue_organ="brain", validation=None).to_df()["uid"]) == ["u1", "u2"]
    assert list(cat.query(tissue_system="Nervous System").to_df()["uid"]) == ["u1", "u2"]
    assert list(cat.query(disease_group="Alzheimer disease", tissue_organ="brain").to_df()["uid"]) == ["u1"]
    assert list(cat.query(disease_class="neurodegenerative disease").to_df()["uid"]) == ["u1"]
    with pytest.raises(ValueError, match="did you mean"):
        cat.query(tissue_organ="brian")
