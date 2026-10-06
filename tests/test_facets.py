from __future__ import annotations

import pytest
from tests._fixtures import make_fixture_catalog
from typer.testing import CliRunner

from sddb.cli import app

runner = CliRunner()


@pytest.fixture
def fixture_catalog_env(tmp_path, monkeypatch):
    df = make_fixture_catalog()
    df.loc[df["uid"] == "uid0003", "technology"] = "VisiumHD"
    p = tmp_path / "cat.parquet"
    df.to_parquet(p)
    monkeypatch.setenv("SDDB_CATALOG_URL", p.as_uri())
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))


def test_facets_lists_columns(fixture_catalog_env):
    r = runner.invoke(app, ["facets"])
    assert r.exit_code == 0, r.output
    assert "technology" in r.output.split()


def test_facets_lists_values(fixture_catalog_env):
    r = runner.invoke(app, ["facets", "technology"])
    assert r.exit_code == 0, r.output
    assert "VisiumHD" in r.output
    assert "Xenium" in r.output


def test_facets_unknown_field(fixture_catalog_env):
    assert runner.invoke(app, ["facets", "nope"]).exit_code == 1


def test_query_zero_match_suggests(fixture_catalog_env):
    r = runner.invoke(app, ["query", "--technology", "Visium HD"])
    assert r.exit_code == 0, r.output
    assert "did you mean" in r.output.lower()
    assert "VisiumHD" in r.output


def test_query_match_no_suggestion(fixture_catalog_env):
    r = runner.invoke(app, ["query", "--technology", "Xenium"])
    assert r.exit_code == 0, r.output
    assert "did you mean" not in r.output.lower()
