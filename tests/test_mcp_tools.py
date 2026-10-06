from __future__ import annotations

import json

import pytest
from tests._fixtures import make_fixture_catalog, make_tiny_sdata_zarr, write_fixture_gene_index

from sddb.mcp import tools


@pytest.fixture
def cat_url(tmp_path):
    zarr = make_tiny_sdata_zarr(tmp_path)
    df = make_fixture_catalog()
    df.loc[df["uid"] == "uid0001", "zarr_url"] = str(zarr)
    p = tmp_path / "cat.parquet"
    df.to_parquet(p)
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    return p.as_uri()


def test_query_tool(cat_url):
    rows = tools.query_tool(cat_url, organism="human")
    assert {r["uid"] for r in rows} == {"uid0001", "uid0002", "uid0005"}
    assert json.loads(json.dumps(rows)) == rows


def test_query_tool_validation(cat_url):
    assert "uid0004" not in {r["uid"] for r in tools.query_tool(cat_url)}
    rows = tools.query_tool(cat_url, validation="all")
    assert "uid0004" in {r["uid"] for r in rows}
    assert {r["uid"] for r in tools.query_tool(cat_url, validation="fail")} == {"uid0004"}


def test_query_tool_truncates(cat_url, monkeypatch):
    monkeypatch.setattr(tools, "MAX_ROWS", 2)
    rows = tools.query_tool(cat_url)
    assert len(rows) == 3
    assert "truncated" in rows[-1]["note"]


def test_describe_tool(cat_url):
    d = tools.describe_tool("uid0001", cat_url)
    assert d["organism"] == "human"
    assert d["elements"]["images/img"]["shape"] == [3, 8, 8]
    assert json.loads(json.dumps(d)) == d


def test_describe_tool_remote_failure_and_missing(cat_url):
    d = tools.describe_tool("uid0002", cat_url)  # s3 url, no network/creds -> metadata only
    assert d["uid"] == "uid0002"
    assert d["elements"] is None
    json.dumps(d)
    with pytest.raises(ValueError, match="not in catalog"):
        tools.describe_tool("nope", cat_url)


def test_genes_tool(cat_url):
    rows = tools.genes_tool("epcam", cat_url)
    assert {r["uid"] for r in rows} == {"uid0001", "uid0002", "uid0005"}
    assert json.loads(json.dumps(rows)) == rows


def test_server_registers_tools():
    pytest.importorskip("mcp")
    import asyncio

    from sddb.mcp.server import build_server

    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert names == {"query", "describe", "genes"}


def test_server_tools_hide_catalog_url():
    pytest.importorskip("mcp")
    import asyncio

    from sddb.mcp.server import build_server

    for t in asyncio.run(build_server().list_tools()):
        assert "catalog_url" not in t.input_schema["properties"]


def test_query_tool_expressing_and_range(cat_url):
    rows = tools.query_tool(cat_url, expressing="EPCAM", n_obs__gte=50_000)
    assert {r["uid"] for r in rows} == {"uid0001", "uid0002"}


def test_query_tool_search(cat_url):
    rows = tools.query_tool(cat_url, organism="human", search="lung")
    assert {r["uid"] for r in rows} == {"uid0001"}


def test_query_tool_expressing_min_fraction(cat_url):
    assert {r["uid"] for r in tools.query_tool(cat_url, expressing="EPCAM")} == {"uid0001", "uid0002", "uid0005"}
    assert {r["uid"] for r in tools.query_tool(cat_url, expressing="EPCAM", min_fraction=0.35)} == {"uid0001"}


def test__query_helper_filters(tmp_path):
    from tests._fixtures import write_fixture_catalog
    from sddb.mcp.tools import _query

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    res = _query(cat.as_uri(), organism="human")
    assert sorted(d.uid for d in res) == ["uid0001", "uid0002", "uid0005"]


def test_facets_tool(tmp_path):
    from tests._fixtures import write_fixture_catalog
    from sddb.mcp.tools import facets_tool
    import pytest

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    cols = facets_tool(catalog_url=cat.as_uri())
    assert "organism" in cols and "technology" in cols
    assert facets_tool("organism", catalog_url=cat.as_uri()) == ["human", "mouse"]
    with pytest.raises(ValueError, match="unknown facet"):
        facets_tool("not_a_facet", catalog_url=cat.as_uri())
