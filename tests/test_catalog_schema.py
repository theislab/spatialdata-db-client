import pandas as pd
import pytest
from tests._fixtures import make_fixture_catalog, make_fixture_gene_index, write_fixture_catalog

from sddb import _catalog_schema as cs


def test_fixture_valid():
    assert cs.validate(make_fixture_catalog()) is None


def test_extra_column_tolerated():
    df = make_fixture_catalog()
    df["brand_new"] = 1
    cs.validate(df)


@pytest.mark.parametrize("col", cs.REQUIRED_COLUMNS)
def test_missing_required(col):
    with pytest.raises(cs.SchemaError, match=col):
        cs.validate(make_fixture_catalog().drop(columns=col))


def test_all_null_required():
    df = make_fixture_catalog()
    df["organism"] = pd.Series([None] * len(df), dtype="string")
    with pytest.raises(cs.SchemaError, match="organism"):
        cs.validate(df)


def test_bad_dtype():
    df = make_fixture_catalog()
    df["n_obs"] = ["a", "b", "c", "d", "e"]
    with pytest.raises(cs.SchemaError, match="n_obs"):
        cs.validate(df)


def test_numpy_dtypes_accepted():
    df = make_fixture_catalog().astype({"n_obs": "int64", "uid": object})
    cs.validate(df)


def test_gene_index():
    gi = make_fixture_gene_index()
    cs.validate(gi, kind="gene_index")
    with pytest.raises(cs.SchemaError, match="symbol"):
        cs.validate(gi.drop(columns="symbol"), kind="gene_index")


def test_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        cs.validate(make_fixture_catalog(), kind="nope")


def test_constants_consistent():
    assert set(cs.FACETS) <= set(cs.CATALOG_COLUMNS)
    assert set(cs.REQUIRED_COLUMNS) <= set(cs.CATALOG_COLUMNS)


def test_parquet_roundtrip(tmp_path):
    p = write_fixture_catalog(tmp_path / "catalog.parquet")
    cs.validate(pd.read_parquet(p))
