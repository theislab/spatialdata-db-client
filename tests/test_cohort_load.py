import inspect

import pandas as pd

from sddb import remote
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset


def test_dataset_load_accepts_elements(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        remote,
        "open_sdata",
        lambda url, lazy=True, elements=None: seen.update(url=url, lazy=lazy, elements=elements) or "SD",
    )
    d = Dataset(pd.Series({"uid": "u1", "zarr_url": "s3://b/1.zarr"}))
    assert d.load(lazy=False, elements=["images/he"]) == "SD"
    assert seen == {"url": "s3://b/1.zarr", "lazy": False, "elements": ["images/he"]}


def test_cohort_load_and_sdatas(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote, "open_sdata", lambda url, lazy=True, elements=None: calls.append((url, lazy, elements)) or url
    )
    coh = SpatialDataCohort(pd.DataFrame({"uid": ["u1", "u2"], "zarr_url": ["s3://b/1.zarr", "s3://b/2.zarr"]}))
    out = coh.load(elements=["tables/t"])
    assert out == ["s3://b/1.zarr", "s3://b/2.zarr"]
    assert all(lazy is False for _, lazy, _ in calls)
    assert all(el == ["tables/t"] for *_, el in calls)
    calls.clear()
    assert coh.sdatas == ["s3://b/1.zarr", "s3://b/2.zarr"]
    assert all(lazy is False and el is None for _, lazy, el in calls)


def test_load_elements_param_present():
    assert "elements" in inspect.signature(Dataset.load).parameters
