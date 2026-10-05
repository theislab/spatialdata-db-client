from __future__ import annotations

import warnings

import pytest
from tests._fixtures import write_fixture_catalog

from sddb._cache import catalog_cache_path, fetch_catalog
from sddb.catalog import Catalog
from sddb.dataset import Results


@pytest.fixture
def cat(tmp_path):
    p = write_fixture_catalog(tmp_path / "cat.parquet")
    return Catalog(p.as_uri(), cache_dir=tmp_path / "cache")


def test_load(cat):
    assert len(cat.to_df()) == 5
    assert isinstance(cat.generated_at, (str, type(None)))


def test_query_organism_and_validation(cat):
    res = cat.query(organism="human")
    assert {d.organism for d in res} == {"human"}
    assert len(res) == 3
    assert "uid0004" not in {d.uid for d in cat.query()}
    assert "uid0004" in {d.uid for d in cat.query(validation=None)}


def test_query_isin_assay_license(cat):
    assert len(cat.query(organism=["human", "mouse"])) == 4
    assert len(cat.query(assay="10x Xenium")) == 2
    assert "uid0004" not in {d.uid for d in cat.query(validation=None, license_set=True)}
    assert len(cat.query(validation=None, license_set=True)) == 4


def test_query_bad_facet(cat):
    with pytest.raises(ValueError, match="organism"):
        cat.query(not_a_facet="x")


def test_search(cat):
    res = cat.search("human xenium")
    assert len(res) == 2
    assert "organism" in res.matched
    assert "technology" in res.matched or "assay" in res.matched
    with pytest.warns(UserWarning, match="nothing matched"):
        none = cat.search("zzzznope")
    assert isinstance(none, Results)
    assert len(none) == 0
    assert none.matched == {}


def test_corrupt_catalog(tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_text("not parquet")
    with pytest.raises(ValueError, match="bad.parquet"):
        Catalog(bad.as_uri(), cache_dir=tmp_path / "c")


def test_fetch_cache(tmp_path):
    src = write_fixture_catalog(tmp_path / "cat.parquet")
    cache = tmp_path / "cache"
    p1 = fetch_catalog(src.as_uri(), cache_dir=cache)
    assert cache in p1.parents
    m = p1.stat().st_mtime_ns
    assert fetch_catalog(src.as_uri(), cache_dir=cache) == p1
    assert p1.stat().st_mtime_ns == m


def test_fetch_unreachable(tmp_path):
    url = "http://127.0.0.1:9/catalog.parquet"
    cache = tmp_path / "cache"
    with pytest.raises(FileNotFoundError):
        fetch_catalog(url, cache_dir=cache)
    seeded = catalog_cache_path(url, cache_dir=cache)
    write_fixture_catalog(seeded)
    with pytest.warns(UserWarning, match="cached catalog"):
        assert fetch_catalog(url, cache_dir=cache) == seeded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(Catalog(url, cache_dir=cache).to_df()) == 5
