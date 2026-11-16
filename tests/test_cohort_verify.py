import os
import urllib.error
import warnings

import pandas as pd
import pytest
from tests._fixtures import write_fixture_catalog

from sddb import cohort as cohort_mod
from sddb._cache import _file_sha256, catalog_cache_path
from sddb.catalog import _ROLLING_SUFFIX, Catalog
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Source
from sddb.manifest import CatalogChangedError, CatalogUnavailableError, CohortDriftError, CohortManifest


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


def _catalog(tmp_path, cache):
    return Catalog(str(write_fixture_catalog(tmp_path / "catalog.parquet")), cache_dir=cache)


def _modify(path):
    """Rewrite the catalog with a changed row and a newer mtime."""
    df = pd.read_parquet(path)
    df.loc[0, "tissue"] = "changed"
    df.to_parquet(path)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def _alive(monkeypatch):
    monkeypatch.setattr(cohort_mod, "_open_fs", lambda url: (_FS({url}), url))


def test_freeze_records_catalog_sha256_and_roundtrips(tmp_path):
    cat = _catalog(tmp_path, tmp_path / "cache")
    man = cat.query(validation=None).freeze()
    assert man.source.sha256 == _file_sha256(tmp_path / "catalog.parquet")
    assert man.source.sha256.startswith("sha256:")
    assert CohortManifest.load(man.write(tmp_path / "c.json")).source.sha256 == man.source.sha256


def test_verify_pinned_catalog_unchanged_passes_changed_raises(tmp_path, monkeypatch):
    _alive(monkeypatch)
    cat = _catalog(tmp_path, tmp_path / "cache")
    coh = cat.query(validation=None)
    coh.verify()  # same bytes: passes
    _modify(tmp_path / "catalog.parquet")
    with pytest.raises(CatalogChangedError) as ei:
        coh.verify()
    err = ei.value
    assert err.url == cat.url
    assert err.expected == coh._source.sha256
    assert err.actual != err.expected
    for part in (err.url, err.expected, err.actual):
        assert part in str(err)
    assert "modified after the cohort was frozen" in str(err)


def test_verify_rolling_catalog_change_warns(tmp_path, monkeypatch):
    _alive(monkeypatch)
    cache = tmp_path / "cache"
    _catalog(tmp_path, cache)
    rolling = "https://example.org/releases/download" + _ROLLING_SUFFIX
    monkeypatch.setattr("sddb._cache.fetch_catalog", lambda url, **_: tmp_path / "catalog.parquet")
    coh = SpatialDataCohort(
        pd.DataFrame({"uid": ["a"], "zarr_url": ["s3://x"]}), source=Source(rolling, cache, sha256="sha256:old")
    )
    with pytest.warns(UserWarning, match="catalog-latest changed"):
        coh.verify()  # warns, does not raise


def test_verify_skips_without_recorded_sha256(tmp_path, monkeypatch):
    _alive(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("catalog fetched")

    monkeypatch.setattr("sddb._cache.fetch_catalog", boom)
    coh = SpatialDataCohort(pd.DataFrame({"uid": ["a"], "zarr_url": ["s3://x"]}), source=Source("gone.parquet"))
    coh.verify()


def test_from_manifest_changed_catalog_raises(tmp_path):
    cat = _catalog(tmp_path, tmp_path / "cache")
    p = cat.query(validation=None).freeze().write(tmp_path / "c.json")
    m = CohortManifest.load(p)
    SpatialDataCohort.from_manifest(m, catalog=cat)  # unchanged: fine
    _modify(tmp_path / "catalog.parquet")
    new = Catalog(str(tmp_path / "catalog.parquet"), cache_dir=tmp_path / "cache")
    with pytest.raises(CatalogChangedError):
        SpatialDataCohort.from_manifest(m, catalog=new)


def test_from_manifest_without_catalog_is_offline(tmp_path, monkeypatch):
    cat = _catalog(tmp_path, tmp_path / "cache")
    m = CohortManifest.load(cat.query(validation=None).freeze().write(tmp_path / "c.json"))
    _modify(tmp_path / "catalog.parquet")

    def boom(*a, **k):
        raise AssertionError("fetched or hashed")

    monkeypatch.setattr("sddb._cache.fetch_catalog", boom)
    monkeypatch.setattr("sddb._cache._file_sha256", boom)
    coh = SpatialDataCohort.from_manifest(m)
    assert coh._source.sha256 == m.source.sha256


def test_verify_raises_when_release_unreachable_but_catalog_falls_back(tmp_path, monkeypatch):
    url = "https://example.org/releases/download/catalog-2026-10-28/catalog.parquet"
    cache = tmp_path / "cache"
    write_fixture_catalog(catalog_cache_path(url, cache_dir=cache))  # a cached copy exists

    def offline(*a, **k):
        raise urllib.error.URLError("network is down")

    monkeypatch.setattr("sddb._cache.urllib.request.urlopen", offline)
    _alive(monkeypatch)
    with pytest.warns(UserWarning, match="using cached catalog"):
        cat = Catalog(url, cache_dir=cache)  # Catalog() still falls back to the cache
    coh = cat.query(validation=None)
    with pytest.raises(CatalogUnavailableError) as ei:
        coh.verify()
    assert ei.value.url == url
    assert url in str(ei.value)
    assert "could not be verified" in str(ei.value)
    assert "network is down" in str(ei.value)


def test_rolling_catalog_records_its_release_and_verify_checks_that_release(tmp_path, monkeypatch):
    """catalog-latest takes its release from sddb_generated_at, and verify() then checks the dated release strictly."""

    import pyarrow.parquet as pq

    _alive(monkeypatch)
    path = write_fixture_catalog(tmp_path / "catalog.parquet")
    table = pq.read_table(path)
    pq.write_table(table.replace_schema_metadata({**table.schema.metadata, b"sddb_generated_at": b"2026-11-01"}), path)
    rolling = "https://example.org/releases/download" + _ROLLING_SUFFIX
    fetched = []
    monkeypatch.setattr("sddb.catalog.fetch_catalog", lambda url, **_: path)
    monkeypatch.setattr("sddb._cache.fetch_catalog", lambda url, **_: (fetched.append(url), path)[1])
    cat = Catalog(rolling)
    assert cat.release == "2026-11-01"
    coh = cat.query(validation=None)
    assert coh._source.release == "2026-11-01"
    coh.verify()
    assert fetched == ["https://example.org/releases/download/catalog-2026-11-01/catalog.parquet"]
    stale = SpatialDataCohort(coh.members, source=Source(rolling, sha256="sha256:old", release="2026-11-01"))
    with pytest.raises(CatalogChangedError):
        stale.verify()  # a changed file for a recorded release raises, where catalog-latest alone only warns
