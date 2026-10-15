import pandas as pd

from sddb.cohort import SpatialDataCohort
from sddb.split import SplitManifest


def _coh(n_studies=6):
    uids = [f"u{i}" for i in range(n_studies)]
    return SpatialDataCohort(
        pd.DataFrame({"uid": uids, "study_id": [f"s{i}" for i in range(n_studies)], "zarr_url": ["x"] * n_studies})
    )


def test_split_returns_sub_cohorts_partitioning_the_parent():
    coh = _coh(6)
    folds = coh.split(by="study_id", train=0.5, val=0.25, test=0.25, seed=0)
    assert set(folds) == {"train", "val", "test"}
    assert all(isinstance(f, SpatialDataCohort) for f in folds.values())
    union = set().union(*[set(f.to_df()["uid"]) for f in folds.values()])
    assert union == set(coh.to_df()["uid"])
    total = sum(len(f) for f in folds.values())
    assert total == len(coh)  # disjoint + covering


def test_split_is_deterministic():
    a = _coh(6).split(by="study_id", train=0.5, val=0.25, test=0.25, seed=7)
    b = _coh(6).split(by="study_id", train=0.5, val=0.25, test=0.25, seed=7)
    assert {k: sorted(v.to_df()["uid"]) for k, v in a.items()} == {k: sorted(v.to_df()["uid"]) for k, v in b.items()}


def test_fold_carries_split_manifest(tmp_path):
    folds = _coh(6).split(by="study_id", train=0.5, val=0.25, test=0.25, seed=0)
    sm = folds["train"].split_manifest
    assert isinstance(sm, SplitManifest)
    assert sm.write(tmp_path / "split.json").exists()
