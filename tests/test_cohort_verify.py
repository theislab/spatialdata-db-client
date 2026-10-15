import warnings

import pandas as pd
import pytest

from sddb import cohort as cohort_mod
from sddb.cohort import SpatialDataCohort
from sddb.manifest import CohortDriftError


class _FS:
    def __init__(self, present):
        self._present = present

    def exists(self, root):
        return root in self._present


def test_verify_passes_when_all_present(monkeypatch):
    present = {"root1", "root2"}
    monkeypatch.setattr(cohort_mod, "_open_fs", lambda url: (_FS(present), url.split("//")[-1]))
    coh = SpatialDataCohort(pd.DataFrame({"uid": ["a", "b"], "zarr_url": ["s3://root1", "s3://root2"]}))
    coh.verify()  # must not raise


def test_verify_raises_naming_missing(monkeypatch):
    monkeypatch.setattr(cohort_mod, "_open_fs", lambda url: (_FS({"root1"}), url.split("//")[-1]))
    coh = SpatialDataCohort(pd.DataFrame({"uid": ["a", "b"], "zarr_url": ["s3://root1", "s3://gone"]}))
    with pytest.raises(CohortDriftError) as ei:
        coh.verify()
    assert "b" in ei.value.drifted


def test_check_updates_detects_superseded():
    # pinned uid has version 0000; catalog's current row for the same stem is 0002
    stem = "s" * 16
    pinned = SpatialDataCohort(pd.DataFrame({"uid": [stem + "0000"], "zarr_url": ["s3://x"]}))

    class _Cat:
        def to_df(self_):
            return pd.DataFrame({"uid": [stem + "0002"], "zarr_url": ["s3://y"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        superseded = pinned.check_updates(_Cat())
    assert superseded == {stem + "0000": stem + "0002"}
    assert any("supersed" in str(x.message).lower() for x in w)
