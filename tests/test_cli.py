from __future__ import annotations

import json

import pytest
from tests._fixtures import make_fixture_catalog, make_tiny_sdata_zarr
from typer.testing import CliRunner

from sddb.cli import app

runner = CliRunner()


@pytest.fixture
def cat_url(tmp_path):
    p = tmp_path / "cat.parquet"
    make_fixture_catalog().to_parquet(p)
    return p.as_uri()


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))


def test_query_table(cat_url):
    r = runner.invoke(app, ["query", "--catalog", cat_url, "--organism", "human"])
    assert r.exit_code == 0, r.output
    assert "uid0001" in r.output
    assert "uid0003" not in r.output
    # Skip the catalog header line (starts with #)
    lines = [l for l in r.output.splitlines() if not l.startswith("#")]
    assert lines[0].split() == ["uid", "technology", "organism", "tissue", "n_obs"]


def test_query_table_renders_na(tmp_path):
    # a pass row with a missing value in a displayed column must render "-" , not crash:
    # pandas pd.NA has an ambiguous truth value, which the real catalog exposes.
    import pandas as pd

    df = make_fixture_catalog()
    df.loc[df["uid"] == "uid0001", "n_obs"] = pd.NA
    p = tmp_path / "na.parquet"
    df.to_parquet(p)
    r = runner.invoke(app, ["query", "--catalog", p.as_uri(), "--organism", "human"])
    assert r.exit_code == 0, r.output
    assert "uid0001" in r.output
    assert " -" in r.output


def test_query_json_and_validation_all(cat_url):
    r = runner.invoke(app, ["query", "--catalog", cat_url, "--json"])
    assert {d["uid"] for d in json.loads(r.output)} == {"uid0001", "uid0002", "uid0003", "uid0005"}
    r = runner.invoke(app, ["query", "--catalog", cat_url, "--json", "--validation", "all"])
    assert len(json.loads(r.output)) == 5


def test_query_bad_validation(cat_url):
    assert runner.invoke(app, ["query", "--catalog", cat_url, "--validation", "x"]).exit_code != 0


def test_viewer_url(cat_url):
    r = runner.invoke(app, ["viewer-url", "uid0001", "--catalog", cat_url])
    assert r.exit_code == 0
    assert r.output.strip().startswith("https://vitessce.io/?url=https://lamin.ai/storage/s3/")
    assert runner.invoke(app, ["viewer-url", "nope", "--catalog", cat_url]).exit_code == 1


def test_download(tmp_path):
    zarr = make_tiny_sdata_zarr(tmp_path)
    df = make_fixture_catalog()
    df["zarr_url"] = str(zarr)
    p = tmp_path / "local.parquet"
    df.to_parquet(p)
    dest = tmp_path / "out"
    r = runner.invoke(app, ["download", "uid0001", "uid0002", "--dest", str(dest), "--catalog", p.as_uri()])
    assert r.exit_code == 0, r.output
    assert r.output.strip() == str(dest / "manifest.json")
    m = json.loads((dest / "manifest.json").read_text())
    assert [e["status"] for e in m["entries"]] == ["complete", "complete"]
    assert (dest / "uid0001.zarr").is_dir()
