from pathlib import Path

import pandas as pd
import pytest
from tests._fixtures import make_fixture_catalog

from sddb import viz
from sddb.dataset import Dataset

SAMPLE = Path(__file__).parent.parent / "docs" / "_data" / "catalog_sample.parquet"


def _real_url() -> str:
    return str(pd.read_parquet(SAMPLE).vitessce_url.dropna().iloc[0])


def test_decode_real_url():
    cfg = viz.config_from_vitessce_url(_real_url())
    assert isinstance(cfg, dict)
    assert "version" in cfg
    assert "datasets" in cfg


def test_decode_roundtrip():
    from urllib.parse import quote

    url = "https://vitessce.io#?edit=false&url=data:," + quote(quote('{"version": "1", "a": "b&c"}'))
    assert viz.config_from_vitessce_url(url) == {"version": "1", "a": "b&c"}


def test_non_inlined_returns_none():
    assert viz.config_from_vitessce_url("s3://bucket/x_vitessce.json") is None
    assert viz.config_from_vitessce_url("https://example.org/c.json") is None


def test_view_interactive_inlined_and_fallback(monkeypatch):
    pytest.importorskip("vitessce")
    seen = []
    monkeypatch.setattr(viz, "render_config", lambda c: seen.append(c) or "w")
    row = make_fixture_catalog().iloc[0].copy()
    row["vitessce_url"] = _real_url()
    assert Dataset(row).view_interactive(mode="config") == "w"
    assert isinstance(seen[0], dict)
    Dataset(make_fixture_catalog().iloc[0]).view_interactive(mode="config")
    assert seen[1].startswith("https://lamin.ai/storage/s3/")
