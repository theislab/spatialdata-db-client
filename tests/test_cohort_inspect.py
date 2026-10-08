from __future__ import annotations

import pandas as pd
from tests._fixtures import make_fixture_catalog

from sddb.cohort import SpatialDataCohort


def _cohort():
    df = make_fixture_catalog()
    df.loc[df["uid"] == "uid0004", "study_id"] = pd.NA  # one unknown study
    return SpatialDataCohort(df)


def test_groupby_includes_unknown_bucket():
    out = _cohort().groupby("study_id")
    assert "unknown" in set(out["study_id"])
    assert int(out.loc[out["study_id"] == "unknown", "n"].iloc[0]) == 1
    assert out["n"].sum() == 5  # every member counted exactly once


def test_modalities_uses_catalog_flags_no_network(monkeypatch):
    import sddb.remote as remote

    def boom(*a, **k):
        raise AssertionError("network!")

    monkeypatch.setattr(remote, "elements", boom)
    out = _cohort().modalities()  # must not touch remote.elements
    assert set(out.columns) == {"modality", "present"}


def test_coverage_counts_present_and_unknown():
    cov = _cohort().coverage()
    row = cov.loc[cov["field"] == "study_id"].iloc[0]
    assert int(row["present"]) == 4
    assert int(row["unknown"]) == 1
