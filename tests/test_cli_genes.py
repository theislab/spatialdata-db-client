from __future__ import annotations

import pandas as pd
import pytest
from tests._fixtures import make_fixture_catalog
from typer.testing import CliRunner

from sddb.catalog import Catalog
from sddb.cli import app

runner = CliRunner()


class _FakeIndex:
    def ranked(self, symbol):
        return pd.DataFrame(
            {"uid": ["uid0002", "uid0001", "uid0003"], "fraction_obs_detected": [0.9, 0.5, 0.1], "total_counts": [10.0, 5.0, 1.0]}
        )


@pytest.fixture
def cat_url(tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(Catalog, "genes", property(lambda self: _FakeIndex()))
    p = tmp_path / "cat.parquet"
    make_fixture_catalog().to_parquet(p)
    return p.as_uri()


def test_genes_lists_uids(cat_url):
    r = runner.invoke(app, ["genes", "EPCAM", "--catalog", cat_url])
    assert r.exit_code == 0, r.output
    assert r.output.index("uid0002") < r.output.index("uid0001")


def test_genes_limit(cat_url):
    r = runner.invoke(app, ["genes", "EPCAM", "--limit", "1", "--catalog", cat_url])
    assert "uid0002" in r.output
    assert "uid0001" not in r.output
