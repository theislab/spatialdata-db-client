from __future__ import annotations

import warnings

import pytest
from tests._fixtures import make_fixture_catalog

from sddb.cohort import SpatialDataCohort


def test_viewer_url():
    d = SpatialDataCohort(make_fixture_catalog())[0]
    assert d.viewer_url() == (
        "https://vitessce.io/?url=https://lamin.ai/storage/s3/scverse-spatial-eu-central-1/.lamindb/uid0001_vitessce.json"
    )


def test_viewer_url_ready_link_unchanged():
    df = make_fixture_catalog().iloc[:1].copy()
    ready = "https://vitessce.io#?edit=false&url=data:,%7B%22version%22%3A%221.0.16%22%7D"
    df["vitessce_url"] = ready
    assert SpatialDataCohort(df)[0].viewer_url() == ready


def test_viewer_missing(monkeypatch):
    df = make_fixture_catalog().iloc[:1].copy()
    df["vitessce_url"] = None
    with pytest.raises(ValueError, match="no vitessce_url"):
        SpatialDataCohort(df)[0].viewer_url()


def test_view_opens(monkeypatch):
    seen = []
    monkeypatch.setattr("webbrowser.open", seen.append)
    SpatialDataCohort(make_fixture_catalog())[0].view()
    assert seen
    assert seen[0].startswith("https://vitessce.io/?url=")


_STORE = "s3://scverse-spatial-eu-central-1/.lamindb/{}.zarr"


def _link(store_id, times=2):
    """A published-style viewer link whose data: config names ``store_id``, URL-encoded ``times`` times."""
    from urllib.parse import quote

    config = f'{{"datasets": [{{"files": [{{"url": "https://lamin.ai/storage/s3/b/.lamindb/{store_id}.zarr"}}]}}]}}'
    for _ in range(times):
        config = quote(config, safe="")
    return "https://vitessce.io#?edit=false&url=data:," + config


def _row(vitessce_url, zarr_id="ownStore00000000"):
    df = make_fixture_catalog().iloc[:1].copy()
    df["vitessce_url"] = vitessce_url
    df["zarr_url"] = _STORE.format(zarr_id)
    return SpatialDataCohort(df)[0]


@pytest.mark.parametrize("times", [1, 2])
def test_loads_other_store_decodes_the_link(times):
    from sddb.dataset import _loads_other_store

    zarr = _STORE.format("ownStore00000000")
    assert not _loads_other_store(_link("ownStore00000000", times), zarr)
    assert _loads_other_store(_link("otherStore000000", times), zarr)
    assert not _loads_other_store(_link("otherStore000000", times), None)  # nothing to compare against
    assert not _loads_other_store("https://vitessce.io/?url=data:,%7B%7D", zarr)  # no store id in the link


def test_viewer_url_keeps_a_matching_link():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _row(_link("ownStore00000000")).viewer_url() == _link("ownStore00000000")


def test_viewer_url_hides_another_datasets_store(monkeypatch):
    d = _row(_link("otherStore000000"))
    with pytest.warns(UserWarning, match="viewer link") as record:
        assert d.viewer_url() is None
    assert [str(w.message) for w in record] == [
        f"the viewer link for {d.uid} loads another dataset's store; it is hidden until the catalog is fixed"
    ]
    seen = []
    monkeypatch.setattr("webbrowser.open", seen.append)
    with pytest.warns(UserWarning, match="another dataset's store"), pytest.raises(ValueError, match="no usable"):
        d.view()
    assert seen == []


def test_view_without_url_raises():
    with pytest.raises(ValueError, match="no vitessce_url"):
        _row(None).view()
