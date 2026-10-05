from __future__ import annotations

import pytest
from tests._fixtures import make_fixture_catalog

from sddb.dataset import Results


def test_viewer_url():
    d = Results(make_fixture_catalog())[0]
    assert d.viewer_url() == (
        "https://vitessce.io/?url=https://lamin.ai/storage/s3/scverse-spatial-eu-central-1/.lamindb/uid0001_vitessce.json"
    )


def test_viewer_url_ready_link_unchanged():
    df = make_fixture_catalog().iloc[:1].copy()
    ready = "https://vitessce.io#?edit=false&url=data:,%7B%22version%22%3A%221.0.16%22%7D"
    df["vitessce_url"] = ready
    assert Results(df)[0].viewer_url() == ready


def test_viewer_missing(monkeypatch):
    df = make_fixture_catalog().iloc[:1].copy()
    df["vitessce_url"] = None
    with pytest.raises(ValueError, match="no vitessce_url"):
        Results(df)[0].viewer_url()


def test_view_opens(monkeypatch):
    seen = []
    monkeypatch.setattr("webbrowser.open", seen.append)
    Results(make_fixture_catalog())[0].view()
    assert seen
    assert seen[0].startswith("https://vitessce.io/?url=")
