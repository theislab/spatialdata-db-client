from __future__ import annotations

import shutil

import pytest
from tests._fixtures import make_fixture_catalog, make_tiny_sdata_zarr

from sddb import Manifest, manifest
from sddb.dataset import Results, Source


@pytest.fixture
def cohort(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    df = make_fixture_catalog().iloc[:2].copy()
    df["zarr_url"] = str(z)
    return Results(df, source=Source(version="v1"))


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


def test_failed_entry_refetched(tmp_path, cohort):
    df = cohort.to_df()
    df.loc[1, "zarr_url"] = str(tmp_path / "missing.zarr")
    m = Results(df).download(tmp_path / "out")
    assert [e.status for e in m.entries] == ["complete", "failed"]
    assert not list((tmp_path / "out").glob("*.part"))


@pytest.mark.network
@pytest.mark.skip(reason="deferred: real-S3 cohort download needs network + disk (Ruling 3)")
def test_real_s3_cohort_download(tmp_path):
    from sddb import Catalog

    res = Catalog().query(technology="Visium")[:1]
    m = res.download(tmp_path)
    assert m.entries[0].status == "complete"
