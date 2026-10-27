import pandas as pd
import pytest

from sddb.catalog import Catalog


def _cat(tmp_path):
    df = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u3"],
            "study_id": ["s_a", "s_a", "s_b"],
            "technology": ["Visium HD"] * 3,
            "assay": ["spatial"] * 3,
            "organism": ["human"] * 3,
            "validation_status": ["pass"] * 3,
            "zarr_url": ["s3://b/1.zarr", "s3://b/2.zarr", "s3://b/3.zarr"],
            "collections": [["c-x", "c-y"], None, ["c-y"]],
        }
    )
    p = tmp_path / "catalog.parquet"
    df.to_parquet(p)
    return Catalog.from_file(p)


def test_cohort_by_uids(tmp_path):
    coh = _cat(tmp_path).cohort(uids=["u1", "u3"])
    assert sorted(coh.to_df()["uid"]) == ["u1", "u3"]


def test_cohort_by_study_ids(tmp_path):
    coh = _cat(tmp_path).cohort(study_ids=["s_a"])
    assert sorted(coh.to_df()["uid"]) == ["u1", "u2"]


def test_cohort_unknown_uid_raises(tmp_path):
    with pytest.raises(KeyError, match="u9"):
        _cat(tmp_path).cohort(uids=["u1", "u9"])


def test_cohort_requires_exactly_one_selector(tmp_path):
    cat = _cat(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        cat.cohort()
    with pytest.raises(ValueError, match="exactly one"):
        cat.cohort(uids=["u1"], study_ids=["s_a"])


def test_collection(tmp_path):
    cat = _cat(tmp_path)
    assert list(cat.collection("c-y").to_df()["uid"]) == ["u1", "u3"]
    assert cat.collection("c-x").freeze().filter == {"collection": "c-x"}
    with pytest.raises(KeyError, match="c-z"):
        cat.collection("c-z")
