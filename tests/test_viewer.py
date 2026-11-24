from __future__ import annotations

import pytest
from tests._fixtures import make_fixture_catalog

from sddb.cohort import SpatialDataCohort

_MFE = "https://www.merfisheyes.com/spatialdata-viewer/from-s3?url="


def _row(zarr_url):
    df = make_fixture_catalog().iloc[:1].copy()
    df["zarr_url"] = zarr_url
    return SpatialDataCohort(df)[0]


def test_viewer_url_is_the_websites_merfisheyes_link():
    d = _row("s3://scverse-spatial-eu-central-1/.lamindb/HgHSRxH91i5EbUZO.zarr")
    # the same string as the website's viewerUrl() (spatialdata-db-web scripts/build-inventory.test.mjs)
    assert d.viewer_url() == (
        _MFE
        + "https%3A%2F%2Fscverse-spatial-eu-central-1.s3.eu-central-1.amazonaws.com%2F.lamindb%2FHgHSRxH91i5EbUZO.zarr"
    )


def test_no_viewer_outside_a_public_bucket(monkeypatch):
    seen = []
    monkeypatch.setattr("webbrowser.open", seen.append)
    for url in ("s3://other-bucket/a.zarr", "/local/a.zarr"):
        assert _row(url).viewer_url() is None
        with pytest.raises(ValueError, match="no viewer link"):
            _row(url).view()
    assert seen == []


def test_view_opens(monkeypatch):
    seen = []
    monkeypatch.setattr("webbrowser.open", seen.append)
    _row("s3://scverse-spatial-eu-central-1/.lamindb/a.zarr").view()
    assert len(seen) == 1
    assert seen[0].startswith(_MFE)
