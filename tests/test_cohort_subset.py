import pandas as pd
import pytest

from sddb.cohort import SpatialDataCohort


def _coh(uids, tissue):
    return SpatialDataCohort(pd.DataFrame({"uid": uids, "tissue": tissue, "zarr_url": ["x"] * len(uids)}))


def test_filter_returns_new_cohort_and_leaves_source_unchanged():
    coh = _coh(["a", "b", "c"], ["breast", "lung", "breast"])
    sub = coh.filter(tissue="breast")
    assert sorted(sub.to_df()["uid"]) == ["a", "c"]
    assert len(coh) == 3  # unchanged


def test_filter_list_value_is_isin():
    coh = _coh(["a", "b", "c"], ["breast", "lung", "brain"])
    assert sorted(coh.filter(tissue=["breast", "brain"]).to_df()["uid"]) == ["a", "c"]


def test_filter_unknown_column_raises():
    with pytest.raises(ValueError, match="nope"):
        _coh(["a"], ["breast"]).filter(nope="x")


def test_where_mask():
    coh = _coh(["a", "b"], ["breast", "lung"])
    out = coh.where(coh.members["tissue"] == "lung")
    assert list(out.to_df()["uid"]) == ["b"]


def test_union_dedupes_by_uid():
    a = _coh(["a", "b"], ["breast", "lung"])
    b = _coh(["b", "c"], ["lung", "brain"])
    u = a | b
    assert sorted(u.to_df()["uid"]) == ["a", "b", "c"]  # b not doubled
    assert len(u) == 3


def test_difference():
    a = _coh(["a", "b", "c"], ["t"] * 3)
    b = _coh(["b"], ["t"])
    assert sorted((a - b).to_df()["uid"]) == ["a", "c"]


def test_empty_filter_is_empty_cohort_not_error():
    coh = _coh(["a"], ["breast"])
    out = coh.filter(tissue="nonexistent")
    assert len(out) == 0
    out.summary()  # must not raise on an empty cohort


def test_slice_preserves_filter():
    coh = SpatialDataCohort(pd.DataFrame({"uid": ["a", "b", "c"], "zarr_url": ["x"] * 3}), filter={"tissue": "breast"})
    assert coh[:2]._filter == {"tissue": "breast"}
