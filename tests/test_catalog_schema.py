import pandas as pd
import pytest
from tests._fixtures import (
    make_fixture_catalog,
    make_fixture_collections,
    make_fixture_gene_index,
    write_fixture_catalog,
    write_fixture_collections,
)

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


def test_collections_fixture_valid():
    assert cs.validate(make_fixture_collections(), kind="collections") is None


def test_collections_missing_required():
    df = make_fixture_collections().drop(columns="members")
    with pytest.raises(cs.SchemaError, match="members"):
        cs.validate(df, kind="collections")


def test_collections_parquet_roundtrip(tmp_path):
    p = write_fixture_collections(tmp_path / "collections.parquet")
    cs.validate(pd.read_parquet(p), kind="collections")


def test_collection_constants_consistent():
    assert set(cs.COLLECTION_REQUIRED) <= set(cs.COLLECTION_COLUMNS)


def test_list_string_rejects_non_strings():
    df = make_fixture_catalog()
    df["collections"] = [[i] for i in range(len(df))]  # list<int>, not list<str>
    with pytest.raises(cs.SchemaError, match="collections"):
        cs.validate(df)


def test_list_string_accepts_scalar_nulls():
    import numpy as np

    df = make_fixture_catalog()
    df["staining_method"] = [["H&E"], np.nan, None, [], ["IF"]]  # NaN/None cells are null, not invalid
    cs.validate(df)


def test_nullable_int_with_null_validates():
    df = make_fixture_catalog()
    df.loc[df.index[0], "n_obs"] = pd.NA
    cs.validate(df)


def test_string_arrowdtype_accepted():
    pa = pytest.importorskip("pyarrow")
    df = make_fixture_catalog()
    df["organism"] = df["organism"].astype(pd.ArrowDtype(pa.string()))
    cs.validate(df)


def test_fixture_is_row_count_agnostic():
    # the fixture derives every column from _ROWS, so adding rows must not raise
    from tests import _fixtures

    orig = _fixtures._ROWS
    _fixtures._ROWS = [*orig, orig[0]]
    try:
        cs.validate(make_fixture_catalog())
    finally:
        _fixtures._ROWS = orig
