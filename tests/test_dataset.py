from __future__ import annotations

import pandas as pd
import pytest
from tests._fixtures import make_fixture_catalog, make_tiny_sdata_zarr

from sddb.dataset import Dataset, Results


def test_results_sequence():
    res = Results(make_fixture_catalog())
    assert len(res) == 5
    assert isinstance(res[0], Dataset)
    sub = res[1:3]
    assert isinstance(sub, Results)
    assert len(sub) == 2
    assert [d.uid for d in res][:2] == ["uid0001", "uid0002"]
    assert repr(res) == "<Results: 5 datasets>"
    assert res.matched == {}
    assert isinstance(res.to_df(), pd.DataFrame)


def test_dataset_attrs():
    d = Results(make_fixture_catalog())[3]
    assert d.uid == "uid0004"
    assert d.organism == "mouse"
    assert d.zarr_url.endswith("uid0004.zarr")
    assert d.tier is None  # NA -> None
    assert d.size_bytes is None  # absent column -> None
    assert "uid0004" in repr(d)
    with pytest.raises(AttributeError):
        _ = d.nonsense


def test_load_and_elements(tmp_path):
    zarr_path = make_tiny_sdata_zarr(tmp_path)
    df = make_fixture_catalog().iloc[:1].copy()
    df["zarr_url"] = str(zarr_path)
    d = Results(df)[0]
    assert "img" in d.load(lazy=True).images
    assert d.elements()["images/img"]["shape"] == (3, 8, 8)
    with pytest.raises(NotImplementedError):
        d.load(version="x")
