import pandas as pd
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
