import warnings

import pandas as pd
import pytest

from sddb import manifest as m
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Source


def _coh(uid, url, source_url="cat.parquet"):
    df = pd.DataFrame({"uid": [uid], "zarr_url": [url]})
    return SpatialDataCohort(df, source=Source(source_url))


def test_strict_raises_on_missing_store(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_store_exists", lambda url: False)
    coh = _coh("u1", "s3://b/missing.zarr")
    with pytest.raises(m.CohortDriftError, match="u1"):
        coh.download(tmp_path, strict=True)


def test_fallback_resolves_latest_with_warning(monkeypatch, tmp_path):
    stem = "s" * 16
    present = {"s3://b/latest.zarr"}
    monkeypatch.setattr(m, "_store_exists", lambda url: url in present)

    def _latest_for(source, stem_arg):
        assert stem_arg == stem
        return stem + "0002", "s3://b/latest.zarr"

    monkeypatch.setattr(m, "_latest_for_stem", _latest_for)
    captured = {}
    monkeypatch.setattr(m, "_fetch_all", lambda pairs, root, cv, workers: captured.setdefault("pairs", pairs))
    coh = _coh(stem + "0000", "s3://b/missing.zarr")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coh.download(tmp_path)
    assert captured["pairs"] == [(stem + "0000", "s3://b/latest.zarr")]
    assert any("fall" in str(x.message).lower() or "latest" in str(x.message).lower() for x in w)


def test_missing_store_no_fallback_raises_nonstrict(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "_store_exists", lambda url: False)
    coh = SpatialDataCohort(pd.DataFrame({"uid": ["u1"], "zarr_url": [str(tmp_path / "nope.zarr")]}))
    with pytest.raises(m.CohortDriftError, match="u1"):
        coh.download(tmp_path / "out")


def test_latest_for_stem_no_url_returns_none():
    assert m._latest_for_stem(Source(), "s" * 16) is None


def test_transient_liveness_error_does_not_fall_back(monkeypatch, tmp_path):
    def boom(url):
        raise OSError("connection reset")

    monkeypatch.setattr(m, "_store_exists", boom)
    monkeypatch.setattr(m, "_latest_for_stem", lambda s, st: pytest.fail("must not fall back"))
    coh = _coh("u1", "s3://b/x.zarr")
    with pytest.raises(RuntimeError, match="could not determine liveness of u1"):
        coh.download(tmp_path)


def test_complete_member_skips_liveness_on_resume(monkeypatch, tmp_path):
    from tests._fixtures import make_tiny_sdata_zarr

    z = make_tiny_sdata_zarr(tmp_path)
    coh = _coh("u1", str(z))
    coh.download(tmp_path / "out")
    monkeypatch.setattr(m, "_store_exists", lambda url: False)
    monkeypatch.setattr(m, "_latest_for_stem", lambda s, st: pytest.fail("must not fall back"))
    res = coh.download(tmp_path / "out", strict=True)
    assert [e.status for e in res.entries] == ["complete"]
