from __future__ import annotations

import shutil

import pytest
from tests._fixtures import make_fixture_catalog, make_tiny_sdata_zarr

from sddb import Manifest, ManifestVersionMismatch, manifest
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Source


@pytest.fixture
def cohort(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    df = make_fixture_catalog().iloc[:2].copy()
    df["zarr_url"] = str(z)
    return SpatialDataCohort(df, source=Source(version="v1"))


def test_download_and_resume(tmp_path, cohort, monkeypatch):
    calls = []
    real = manifest._copy_store
    monkeypatch.setattr(manifest, "_copy_store", lambda u, d: (calls.append(d.name), real(u, d))[1])
    dest = tmp_path / "out"
    m = cohort.download(dest, pin_versions=True)
    assert len(calls) == 2
    assert [e.status for e in m.entries] == ["complete", "complete"]
    assert (dest / "uid0001.zarr").is_dir()
    assert m.catalog_version == "v1"
    assert all(e.sha256 and e.size_bytes for e in m.entries)
    assert Manifest.read(dest / "manifest.json") == m

    cohort.download(dest)  # re-run: nothing re-copied
    assert len(calls) == 2

    shutil.rmtree(dest / "uid0002.zarr")  # a missing store is re-fetched, the other is not
    cohort.download(dest)
    assert calls[2:] == ["uid0002.zarr.part"]


def test_failed_entry_refetched(tmp_path, cohort, monkeypatch):
    # store exists (passes liveness) but the copy fails for one member -> recorded "failed", refetched later
    real = manifest._copy_store
    bad = str(cohort.to_df()["uid"].iloc[1]) + ".zarr.part"

    def flaky(u, d):
        if d.name == bad:
            raise OSError("transient copy failure")
        return real(u, d)

    monkeypatch.setattr(manifest, "_copy_store", flaky)
    dest = tmp_path / "out"
    m = cohort.download(dest)
    assert [e.status for e in m.entries] == ["complete", "failed"]
    assert not list(dest.glob("*.part"))
    monkeypatch.setattr(manifest, "_copy_store", real)
    m2 = cohort.download(dest)
    assert [e.status for e in m2.entries] == ["complete", "complete"]


def test_version_mismatch_and_allow(tmp_path, cohort):
    dest = tmp_path / "out"
    cohort.download(dest, pin_versions=True)  # pinned to "v1"
    other = SpatialDataCohort(cohort.to_df(), source=Source(version="B"))
    with pytest.raises(ManifestVersionMismatch, match="'v1'.*'B'"):
        other.download(dest)
    m = other.download(dest, allow_version_change=True, pin_versions=True)
    assert m.catalog_version == "B"
    assert Manifest.read(dest / "manifest.json").catalog_version == "B"


def test_refetch_reproduces_stores(tmp_path, cohort):
    m = cohort.download(tmp_path / "a", pin_versions=True)
    again = Manifest.read(tmp_path / "a" / "manifest.json").refetch(tmp_path / "b")
    assert again.catalog_version == "v1"
    assert [(e.uid, e.zarr_url, e.sha256) for e in again.entries] == [(e.uid, e.zarr_url, e.sha256) for e in m.entries]
    assert (tmp_path / "b" / "uid0002.zarr").is_dir()


@pytest.mark.network
@pytest.mark.skip(reason="deferred: real-S3 cohort download needs network + disk (Ruling 3)")
def test_real_s3_cohort_download(tmp_path):
    from sddb import Catalog

    res = Catalog().query(technology="Visium")[:1]
    m = res.download(tmp_path)
    assert m.entries[0].status == "complete"


def test_failed_entry_records_error(tmp_path, cohort, monkeypatch):
    real = manifest._copy_store

    def flaky(url, dest):
        if dest.name.startswith("uid0002"):
            raise OSError("boom")
        real(url, dest)

    monkeypatch.setattr(manifest, "_copy_store", flaky)
    m = cohort.download(tmp_path / "out")
    ok, bad = m.entries
    assert (ok.status, ok.error) == ("complete", None)
    assert bad.status == "failed"
    assert bad.error == "OSError: boom"
    assert Manifest.read(tmp_path / "out" / "manifest.json") == m


def test_plan_sizes_local_store(tmp_path):
    import pandas as pd
    from tests._fixtures import make_tiny_sdata_zarr

    from sddb.cohort import SpatialDataCohort
    from sddb.manifest import plan_sizes

    store = make_tiny_sdata_zarr(tmp_path)
    res = SpatialDataCohort(pd.DataFrame({"uid": ["uidX"], "zarr_url": [store.as_uri()]}))
    sizes = plan_sizes(res)
    assert len(sizes) == 1
    assert sizes[0][0] == "uidX"
    assert sizes[0][1] > 0


def test_unpinned_download_records_the_release_without_pinning(tmp_path, cohort):
    """catalog_release is written from an unpinned catalog, and a rerun after a new release still works."""
    d = tmp_path / "out"
    SpatialDataCohort(cohort.members, source=Source(release="2026-11-01")).download(d)
    m = Manifest.read(d / "manifest.json")
    assert (m.catalog_version, m.catalog_release) == (None, "2026-11-01")
    SpatialDataCohort(cohort.members, source=Source(release="2026-11-08")).download(d)  # no ManifestVersionMismatch
    assert Manifest.read(d / "manifest.json").catalog_release == "2026-11-08"
