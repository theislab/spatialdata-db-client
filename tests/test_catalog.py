from __future__ import annotations

import warnings

import pytest
from tests._fixtures import make_fixture_catalog, write_fixture_catalog, write_fixture_gene_index

from sddb._cache import catalog_cache_path, fetch_catalog
from sddb.catalog import Catalog
from sddb.cohort import SpatialDataCohort


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


def test_query_by_license_and_noncommercial(cat):
    # exact license id via the facet
    assert {d.uid for d in cat.query(license="CC-BY-NC-SA-4.0")} == {"uid0002"}
    # the headline filter: exclude NonCommercial (uid0002 is CC-BY-NC-SA)
    comm = {d.uid for d in cat.query(validation=None, noncommercial=False)}
    assert "uid0002" not in comm
    assert "uid0001" in comm
    # keep only NonCommercial
    assert {d.uid for d in cat.query(noncommercial=True)} == {"uid0002"}


def test_query_commercial_use_requires_known_license(cat):
    # uid0004 has an unknown license (license_noncommercial False, license_unknown True): not cleared
    comm = {d.uid for d in cat.query(validation=None, noncommercial=False)}
    assert comm == {"uid0001", "uid0003", "uid0005"}
    assert comm == {d.uid for d in cat.query(validation=None, noncommercial=False, license_set=True)}
    # None and True are unchanged: unknown licenses stay in None, and are not NonCommercial
    assert "uid0004" in {d.uid for d in cat.query(validation=None)}
    assert "uid0004" not in {d.uid for d in cat.query(validation=None, noncommercial=True)}


def test_query_tuple_and_set_values(cat):
    assert len(cat.query(organism=("human", "mouse"))) == 4
    assert len(cat.query(organism={"human"})) == 3


def test_query_facet_missing_from_catalog(cat):
    cat._df = cat._df.drop(columns=["tissue"])
    with pytest.raises(ValueError, match="tissue"):
        cat.query(tissue="lung")


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
    assert isinstance(none, SpatialDataCohort)
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
    p1.write_bytes(b"sentinel")
    p1.touch()
    # no-refresh keeps the cached (sentinel) file; mtimes differ from source so allow either, pin via refresh below
    import os

    os.utime(p1, ns=(src.stat().st_mtime_ns, src.stat().st_mtime_ns))
    assert fetch_catalog(src.as_uri(), cache_dir=cache) == p1
    assert p1.read_bytes() == b"sentinel"
    fetch_catalog(src.as_uri(), cache_dir=cache, refresh=True)
    assert p1.read_bytes() == src.read_bytes()


def test_http_last_modified_conditional(tmp_path):
    import http.server
    import threading

    src = write_fixture_catalog(tmp_path / "cat.parquet")
    body = src.read_bytes()
    seen: list[dict[str, str | None]] = []
    stamp = "Wed, 01 Jan 2025 00:00:00 GMT"

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append({"ims": self.headers.get("If-Modified-Since"), "inm": self.headers.get("If-None-Match")})
            if self.headers.get("If-Modified-Since") == stamp:
                self.send_response(304)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Last-Modified", stamp)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/catalog.parquet"
        cache = tmp_path / "cache"
        p = fetch_catalog(url, cache_dir=cache)
        p.write_bytes(b"sentinel")
        assert fetch_catalog(url, cache_dir=cache) == p
        assert p.read_bytes() == b"sentinel"  # 304: not re-downloaded
        assert seen[1] == {"ims": stamp, "inm": None}
        fetch_catalog(url, cache_dir=cache, refresh=True)
        assert p.read_bytes() == body
        assert seen[2] == {"ims": None, "inm": None}
    finally:
        server.shutdown()


def test_search_word_boundary(cat):
    import pandas as pd

    df = cat._df.copy()
    df.loc[0, "tissue"] = "keratinocyte"
    df.loc[1, "organism"] = "rat"
    cat._df = df
    res = cat.search("rat")
    assert res.matched == {"organism": ["rat"]}
    assert "keratinocyte" not in res.matched.get("tissue", [])
    assert isinstance(res.to_df(), pd.DataFrame)


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


def test_env_catalog_url(tmp_path, monkeypatch):
    p = write_fixture_catalog(tmp_path / "env.parquet")
    monkeypatch.setenv("SDDB_CATALOG_URL", p.as_uri())
    assert len(Catalog(cache_dir=tmp_path / "c1")) == 5
    other = write_fixture_catalog(tmp_path / "other.parquet")
    assert Catalog(other.as_uri(), cache_dir=tmp_path / "c2").url == other.as_uri()  # explicit wins


def test_from_file_abs_rel_and_url(tmp_path, monkeypatch):
    p = write_fixture_catalog(tmp_path / "cat.parquet")
    cache = tmp_path / "cache"
    c = Catalog.from_file(p, cache_dir=cache)
    assert len(c.query(organism="human")) == 3
    assert len(Catalog(url=str(p), cache_dir=cache)) == 5
    assert len(Catalog.from_file(p.as_uri(), cache_dir=cache)) == 5
    monkeypatch.chdir(tmp_path)
    rel = Catalog.from_file("cat.parquet", cache_dir=cache)
    assert len(rel.query(organism="human")) == 3
    assert rel.url == str(p.resolve())


@pytest.mark.network
def test_default_catalog_loads_from_published_release(tmp_path):
    """A bare Catalog() fetches the published release asset anonymously (no account, no token)."""
    from sddb.catalog import DEFAULT_CATALOG_URL

    assert DEFAULT_CATALOG_URL.startswith("https://github.com/theislab/spatialdata-db-client/releases/")
    df = Catalog(cache_dir=tmp_path / "cache", refresh=True).to_df()
    assert len(df) > 0
    assert df["zarr_url"].notna().all()
    assert set(df["validation_status"]) <= {"pass", "fail"}


def test_from_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        Catalog.from_file(tmp_path / "nope.parquet", cache_dir=tmp_path / "c")


def test_query_range_gte(cat):
    assert sorted(d.uid for d in cat.query(n_obs__gte=50_000)) == ["uid0001", "uid0002"]


def test_query_range_gt_excludes_boundary(cat):
    # uid0003=4000 excluded by gt; uid0004 fails validation
    assert sorted(d.uid for d in cat.query(n_obs__gt=4_000)) == ["uid0001", "uid0002", "uid0005"]


def test_query_range_lte_and_lt(cat):
    assert sorted(d.uid for d in cat.query(n_features__lte=5_000)) == ["uid0001", "uid0002"]
    assert sorted(d.uid for d in cat.query(total_counts__lt=500_000)) == ["uid0003", "uid0005"]


def test_query_facet_and_range(cat):
    assert sorted(d.uid for d in cat.query(organism="human", n_obs__gte=50_000)) == ["uid0001", "uid0002"]


def test_query_range_unknown_column_errors(cat):
    with pytest.raises(ValueError, match="range column"):
        cat.query(organism__gte=1)


def test_query_range_nonnumeric_value_errors(cat):
    with pytest.raises(ValueError, match="numeric"):
        cat.query(n_obs__gte="big")


def _cat_genes(tmp_path):
    p = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    return Catalog(p.as_uri(), cache_dir=tmp_path / "cache")


def test_query_search_intersects(tmp_path):
    cat = _cat_genes(tmp_path)
    res = cat.query(organism="human", search="lung")
    assert [d.uid for d in res] == ["uid0001"]
    assert res.matched  # search's matched dict is carried through


def test_query_expressing(tmp_path):
    cat = _cat_genes(tmp_path)
    assert sorted(d.uid for d in cat.query(organism="human", expressing="EPCAM")) == ["uid0001", "uid0002", "uid0005"]


def test_query_expressing_multi_gene(tmp_path):
    cat = _cat_genes(tmp_path)
    assert [d.uid for d in cat.query(expressing=["EPCAM", "KRT8"])] == ["uid0001"]
    assert sorted(d.uid for d in cat.query(expressing=["KRT8", "PTPRC"], mode="any")) == ["uid0001", "uid0002"]
    assert [d.uid for d in cat.query(expressing=["ENSG00000119888", "krt8"])] == ["uid0001"]
    assert cat.query(expressing=["EPCAM", "KRT8"], mode="any")._filter["mode"] == "any"


def test_query_expressing_validation_mask_not_doubled(tmp_path):
    cat = _cat_genes(tmp_path)
    assert len(cat.query(expressing=["Alb"])) == 0  # default validation="pass" drops the failing row
    assert [d.uid for d in cat.query(expressing=["Alb"], validation=None)] == ["uid0004"]


def test_query_bad_mode_errors(tmp_path):
    cat = _cat_genes(tmp_path)
    with pytest.raises(ValueError, match="mode must be"):
        cat.query(mode="anyy")


def test_query_expressing_min_fraction(tmp_path):
    cat = _cat_genes(tmp_path)
    assert [d.uid for d in cat.query(expressing="EPCAM", min_fraction=0.35)] == ["uid0001"]


def test_query_all_combined(tmp_path):
    cat = _cat_genes(tmp_path)
    res = cat.query(organism="human", n_obs__gte=50_000, expressing="EPCAM", search="lung")
    assert [d.uid for d in res] == ["uid0001"]


def test_query_min_fraction_without_expressing_errors(tmp_path):
    cat = _cat_genes(tmp_path)
    with pytest.raises(ValueError, match="min_fraction requires expressing"):
        cat.query(min_fraction=0.5)


def test_query_expressing_absent_symbol_empty(tmp_path):
    cat = _cat_genes(tmp_path)
    assert len(cat.query(expressing="NOPE")) == 0


def test_query_search_nothing_empty(tmp_path):
    cat = _cat_genes(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(cat.query(organism="human", search="zzzznotathing")) == 0


def test_query_expressing_keeps_matched(tmp_path):
    cat = _cat_genes(tmp_path)
    assert cat.query(organism="human", expressing="EPCAM", search="lung").matched


def test_query_expressing_empty_skips_gene_index(tmp_path):
    p = write_fixture_catalog(tmp_path / "catalog.parquet")  # deliberately no gene_index.parquet
    cat = Catalog(p.as_uri(), cache_dir=tmp_path / "cache")
    assert len(cat.query(organism="mouse", tissue="lung", expressing="EPCAM")) == 0  # known values, no overlap


@pytest.mark.parametrize("val", [[1, 2], (1, 2), {1}])
def test_query_range_nonscalar_value_errors(cat, val):
    with pytest.raises(ValueError, match="scalar"):
        cat.query(n_obs__gte=val)


def test_query_range_column_absent_from_catalog(tmp_path):
    df = make_fixture_catalog().drop(columns=["total_counts"])
    p = tmp_path / "c.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="not present in this catalog"):
        Catalog(p.as_uri(), cache_dir=tmp_path / "cache").query(total_counts__gte=1)


def test_search_ignores_doi_fragments(cat):
    df = cat._df.copy()
    df["doi"] = "10.1038/s41592-000"
    cat._df = df
    with pytest.warns(UserWarning, match="nothing matched"):
        res = cat.search("10")
    assert "doi" not in res.matched


_REL = "https://github.com/theislab/spatialdata-db-client/releases/download"


def test_default_url_is_rolling_release():
    from sddb.catalog import DEFAULT_CATALOG_URL
    from sddb.citations import DEFAULT_BIB_URL

    assert DEFAULT_CATALOG_URL.endswith("/releases/download/catalog-latest/catalog.parquet")
    assert "releases/latest" not in DEFAULT_CATALOG_URL
    assert DEFAULT_BIB_URL.endswith("/releases/download/catalog-latest/citations.bib")
    assert "releases/latest" not in DEFAULT_BIB_URL


@pytest.fixture
def no_fetch(monkeypatch, tmp_path):
    """Stub the network fetch: the constructor downloads, so serve a local fixture instead."""
    import sddb.catalog as m

    p = write_fixture_catalog(tmp_path / "fixture.parquet")
    monkeypatch.setattr(m, "fetch_catalog", lambda url, **kw: p)


def test_version_pins_dated_release(monkeypatch, no_fetch):
    monkeypatch.delenv("SDDB_CATALOG_URL", raising=False)
    assert Catalog(version="2026-10-16").url == f"{_REL}/catalog-2026-10-16/catalog.parquet"


def test_version_ignored_with_explicit_url(no_fetch):
    with pytest.warns(UserWarning, match="ignored"):
        cat = Catalog("file:///x/catalog.parquet", version="2026-10-16")
    assert cat.url == "file:///x/catalog.parquet"


def test_version_with_custom_env_url_unchanged(monkeypatch, no_fetch):
    monkeypatch.setenv("SDDB_CATALOG_URL", "https://example.org/c.parquet")
    with pytest.warns(UserWarning, match="ignored"):
        cat = Catalog(version="2026-10-16")
    assert cat.url == "https://example.org/c.parquet"


def test_version_pinned_siblings_share_release_dir(monkeypatch, no_fetch):
    from sddb.genes import sibling_url

    monkeypatch.delenv("SDDB_CATALOG_URL", raising=False)
    url = Catalog(version="2026-10-16").url
    assert sibling_url(url, "gene_index.parquet") == f"{_REL}/catalog-2026-10-16/gene_index.parquet"


def test_list_valued_modalities_facet(tmp_path):
    df = make_fixture_catalog()
    df["modalities"] = [["rna", "protein"], ["rna"], ["rna", "msi"], None, ["rna"]]
    p = tmp_path / "cat.parquet"
    df.to_parquet(p)
    cat = Catalog(p.as_uri(), cache_dir=tmp_path / "cache")
    from sddb.catalog import facet_values

    assert facet_values(cat.to_df(), "modalities") == ["msi", "protein", "rna"]
    assert {d.uid for d in cat.query(modalities="msi")} == {"uid0003"}
    assert {d.uid for d in cat.query(modalities=["protein", "msi"])} == {"uid0001", "uid0003"}
    assert len(cat.query(organism="human")) == 3  # scalar facets unchanged
    assert "uid0001" in {d.uid for d in cat.search("human lung")}  # search no longer crashes


def _uid_catalog(tmp_path, uids):
    df = make_fixture_catalog().iloc[: len(uids)].copy()
    df["uid"] = uids
    p = tmp_path / "uids.parquet"
    df.to_parquet(p)
    return Catalog.from_file(p)


def test_dataset_by_stem(tmp_path):
    cat = _uid_catalog(tmp_path, ["AAAAAAAAAAAAAAAA0000", "BBBBBBBBBBBBBBBB0001"])
    assert cat.dataset("AAAAAAAAAAAAAAAA").uid == "AAAAAAAAAAAAAAAA0000"
    assert cat.dataset("BBBBBBBBBBBBBBBB0001").uid == "BBBBBBBBBBBBBBBB0001"
    with pytest.raises(KeyError, match="uid not in catalog"):
        cat.dataset("ZZZZZZZZZZZZZZZZ")


def test_dataset_ambiguous_stem(tmp_path):
    cat = _uid_catalog(tmp_path, ["CCCCCCCCCCCCCCCC0000", "CCCCCCCCCCCCCCCC0001"])
    with pytest.raises(KeyError, match="CCCCCCCCCCCCCCCC0000.*CCCCCCCCCCCCCCCC0001"):
        cat.dataset("CCCCCCCCCCCCCCCC")
    assert cat.dataset("CCCCCCCCCCCCCCCC0001").uid == "CCCCCCCCCCCCCCCC0001"


def _liver_catalog(cat):
    df = cat._df.copy()
    df.loc[df["uid"] == "uid0005", "tissue"] = "liver"  # liver as a tissue (uid0004 is liver too, but fails validation)
    df.loc[df["uid"] == "uid0002", "study_id"] = "geo:GSE1-liver"  # liver only in a study_id slug
    cat._df = df
    return cat


def test_search_token_matches_any_facet(cat):
    res = _liver_catalog(cat).search("liver")
    assert {d.uid for d in res} == {"uid0002", "uid0005"}  # tissue OR study_id, not both
    assert res.matched["tissue"] == ["liver"]
    assert res.matched["study_id"] == ["geo:GSE1-liver"]
    assert {d.uid for d in cat.search("liver")} >= {d.uid for d in cat.query(tissue="liver")}


def test_search_tokens_are_anded(cat):
    assert {d.uid for d in cat.search("breast xenium")} == {"uid0002"}
    assert {d.uid for d in _liver_catalog(cat).search("liver xenium")} == {"uid0002"}
    assert {d.uid for d in cat.search("lung breast")} == set()  # no row is both


def test_search_ignores_unmatched_token(cat):
    with pytest.warns(UserWarning, match=r"\['qqqqzz'\] match no facet value") as rec:
        res = cat.search("lung qqqqzz")
    assert len(rec) == 1
    assert {d.uid for d in res} == {"uid0001"}
    assert res.matched == {"tissue": ["lung"]}


def test_summary_counts_study_ids_without_nulls(cat):
    df = cat._df.copy()
    df["study_id"] = df["study_id"].astype("string")
    df.loc[df["uid"].isin(["uid0004", "uid0005"]), "study_id"] = None
    cat._df = df
    s = cat.query(validation=None).summary()
    assert s["n_objects"] == 5
    assert s["n_studies"] == 2  # Smith2023 and Lee2022; the two rows without a study_id are not a study
    assert s["n_without_study_id"] == 2
