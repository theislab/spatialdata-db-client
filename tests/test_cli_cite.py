from __future__ import annotations

import pytest
from tests._fixtures import make_fixture_catalog, make_fixture_citations_bib
from typer.testing import CliRunner

from sddb.cli import app

runner = CliRunner()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    cat = tmp_path / "cat.parquet"
    make_fixture_catalog().to_parquet(cat)
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    return cat.as_uri(), str(bib)


def _cite(paths, out, *args):
    cat, bib = paths
    return runner.invoke(app, ["cite", *args, "--catalog", cat, "--bib-url", bib, "-o", str(out)])


def test_cite_facet(paths, tmp_path):
    out = tmp_path / "out.bib"
    r = _cite(paths, out, "--technology", "Xenium")
    assert r.exit_code == 0, r.output
    text = out.read_text()
    assert "Smith2023" in text
    assert "Lee2022" not in text
    assert "wrote 1 citations" in r.output


def test_cite_search(paths, tmp_path):
    out = tmp_path / "out.bib"
    r = _cite(paths, out, "--search", "xenium")
    assert r.exit_code == 0, r.output
    assert "Smith2023" in out.read_text()


def test_cite_empty_cohort(paths, tmp_path):
    out = tmp_path / "out.bib"
    r = _cite(paths, out, "--organism", "nosuch")
    assert r.exit_code != 0
    assert "no datasets matched" in r.output
    assert not out.exists()
