from __future__ import annotations

import pandas as pd
import pytest
from tests._fixtures import make_fixture_catalog
from typer.testing import CliRunner

from sddb.catalog import Catalog, facet_values
from sddb.cli import app
from sddb.mcp import tools

runner = CliRunner()


@pytest.fixture
def cat_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    df = make_fixture_catalog()
    df["study_id"] = pd.array(["geo:GSM1", "GEO:GSM2", "hest:TENX1", None, "10x:Xenium_lung"], dtype="string")
    df["hne_image"] = pd.array([True, False, None, True, False], dtype="boolean")
    df["if_image"] = pd.array([None, True, None, None, False], dtype="boolean")
    p = tmp_path / "cat.parquet"
    df.to_parquet(p)
    return p


@pytest.fixture
def cat(cat_path, tmp_path):
    return Catalog(cat_path.as_uri(), cache_dir=tmp_path / "cache")


def uids(cohort):
    return {d.uid for d in cohort}


def test_hne_image_true_false_null(cat):
    assert uids(cat.query(validation=None, hne_image=True)) == {"uid0001", "uid0004"}
    assert uids(cat.query(validation=None, hne_image=False)) == {"uid0002", "uid0005"}
    # uid0003 has hne_image null (unknown): in neither result
    assert uids(cat.query(validation=None, if_image=False)) == {"uid0005"}
    with pytest.raises(ValueError, match="True or False"):
        cat.query(hne_image="yes")


def test_source_derived_and_filterable(cat):
    df = cat.to_df()
    assert df["source"].tolist()[:3] == ["geo", "geo", "hest"]
    assert pd.isna(df["source"].iloc[3])
    assert facet_values(cat.to_df(), "source") == ["10x", "geo", "hest"]
    assert uids(cat.query(source="geo")) == {"uid0001", "uid0002"}
    assert uids(cat.query(source="GEO")) == {"uid0001", "uid0002"}  # case-insensitive alias
    assert uids(cat.query(source=["hest", "10x"])) == {"uid0003", "uid0005"}


def test_redistributable(cat):
    # fixture: uid0004 has an unknown license (license_redistributable False, license_unknown True)
    assert uids(cat.query(validation=None, redistributable=True)) == {"uid0001", "uid0002", "uid0003", "uid0005"}
    assert uids(cat.query(validation=None, redistributable=False)) == {"uid0004"}
    assert cat.query(redistributable=True)._filter["redistributable"] is True


def test_redistributable_requires_known_license(cat):
    cat._df.loc[cat._df["uid"] == "uid0004", "license_redistributable"] = True  # inconsistent row
    assert "uid0004" not in uids(cat.query(validation=None, redistributable=True))


def test_assay_label_resolves_to_technology(cat):
    assert uids(cat.query(technology="10x Xenium")) == {"uid0001", "uid0002"}
    assert uids(cat.query(technology="visium spatial gene-expression")) == {"uid0003", "uid0005"}
    assert uids(cat.query(technology=["xenium", "Visium"])) == {"uid0001", "uid0002", "uid0003", "uid0005"}
    assert uids(cat.query(organism="Human")) == {"uid0001", "uid0002", "uid0005"}


def test_assay_shared_by_two_technologies_resolves_to_both(cat):
    cat._df.loc[cat._df["uid"] == "uid0002", "technology"] = "Xenium2"
    assert uids(cat.query(technology="10x Xenium")) == {"uid0001", "uid0002"}


def test_unknown_value_raises_with_suggestion(cat):
    with pytest.raises(ValueError, match=r"unknown technology value 'Xenum'; did you mean 'Xenium'\?"):
        cat.query(technology="Xenum")
    with pytest.raises(ValueError, match="did you mean 'mouse'"):
        cat.query(organism=["human", "mouze"])
    with pytest.raises(ValueError, match=r"list the values with facets\('tissue'\)"):
        cat.query(tissue="qqqqqqqq")


def test_known_value_without_match_is_empty(cat):
    assert len(cat.query(organism="mouse", tissue="lung")) == 0
    assert len(cat.query(source="hest", technology="Xenium")) == 0


def test_search_ignores_boolean_and_source_facets(cat):
    assert "hne_image" not in (cat.search("true xenium").matched or {})


def test_cli_new_flags(cat_path):
    url = cat_path.as_uri()
    r = runner.invoke(app, ["query", "--catalog", url, "--validation", "all", "--hne-image"])
    assert r.exit_code == 0, r.output
    assert "uid0001" in r.output
    assert "uid0004" in r.output
    assert "uid0002" not in r.output
    r = runner.invoke(app, ["query", "--catalog", url, "--validation", "all", "--no-hne-image"])
    assert "uid0002" in r.output
    assert "uid0001" not in r.output
    r = runner.invoke(app, ["query", "--catalog", url, "--source", "hest"])
    assert "uid0003" in r.output
    assert "uid0001" not in r.output
    r = runner.invoke(app, ["query", "--catalog", url, "--validation", "all", "--no-redistributable"])
    assert "uid0004" in r.output
    assert "uid0001" not in r.output
    r = runner.invoke(app, ["query", "--catalog", url, "--technology", "10x Xenium"])
    assert "uid0001" in r.output
    assert "uid0003" not in r.output
    r = runner.invoke(app, ["query", "--catalog", url, "--source", "gieo"])
    assert r.exit_code == 1
    assert "did you mean 'geo'?" in r.output


def test_mcp_query_tool_new_filters(cat_path):
    url = cat_path.as_uri()
    assert {r["uid"] for r in tools.query_tool(url, source="geo", hne_image=True)} == {"uid0001"}
    assert {r["uid"] for r in tools.query_tool(url, technology="10x Xenium", redistributable=True)} == {
        "uid0001",
        "uid0002",
    }


def test_site_filter_facets(cat):
    df = cat._df.copy()
    df["preservation_method"] = pd.array(["FFPE", "fresh frozen", "FFPE", None, "fresh frozen"], dtype="string")
    df["magnification"] = pd.array(["20x", None, "40x", None, "20x"], dtype="string")
    cat._df = df
    assert facet_values(cat.to_df(), "preservation_method") == ["FFPE", "fresh frozen"]
    assert uids(cat.query(preservation_method="FFPE")) == {"uid0001", "uid0003"}
    assert uids(cat.query(preservation_method="ffpe")) == {"uid0001", "uid0003"}  # alias
    assert uids(cat.query(preservation_method="Fresh-Frozen")) == {"uid0002", "uid0005"}  # alias
    with pytest.raises(ValueError, match="did you mean 'FFPE'"):
        cat.query(preservation_method="FFPEE")
    assert uids(cat.query(magnification="20X")) == {"uid0001", "uid0005"}
    assert uids(cat.query(collections="c-1")) == {"uid0002"}
    with pytest.raises(ValueError, match="unknown collections value 'c-9'"):
        cat.query(collections="c-9")
    assert uids(cat.search("ffpe")) == {"uid0001", "uid0003"}


def test_mcp_query_tool_preservation_method(cat_path):
    df = pd.read_parquet(cat_path)
    df["preservation_method"] = pd.array(["FFPE", None, None, None, "FFPE"], dtype="string")
    df.to_parquet(cat_path)
    assert {r["uid"] for r in tools.query_tool(cat_path.as_uri(), preservation_method="ffpe")} == {"uid0001", "uid0005"}


def test_cli_preservation_method(cat_path):
    df = pd.read_parquet(cat_path)
    df["preservation_method"] = pd.array(["FFPE", None, None, None, "fresh frozen"], dtype="string")
    df.to_parquet(cat_path)
    r = runner.invoke(app, ["query", "--catalog", cat_path.as_uri(), "--preservation-method", "ffpe"])
    assert r.exit_code == 0, r.output
    assert "uid0001" in r.output
    assert "uid0005" not in r.output
