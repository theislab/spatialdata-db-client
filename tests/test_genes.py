from __future__ import annotations

from tests._fixtures import write_fixture_catalog, write_fixture_gene_index

from sddb import Catalog


def _cat(tmp_path):
    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    return Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache")


def test_datasets_with(tmp_path):
    cat = _cat(tmp_path)
    res = cat.genes.datasets_with("epcam")
    assert sorted(d.uid for d in res) == ["uid0001", "uid0002", "uid0005"]
    assert len(cat.genes.datasets_with("NOPE")) == 0
    assert [d.uid for d in cat.genes.datasets_with("EPCAM", min_fraction=0.3)] == ["uid0001", "uid0002"]
    assert cat.genes is cat.genes


def test_datasets_with_validation(tmp_path):
    import pandas as pd
    from tests._fixtures import make_fixture_gene_index

    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    gi = make_fixture_gene_index()
    extra = gi.iloc[[6]].copy()  # Alb in uid0004 (validation fail); add it to pass uid0003 too
    extra["uid"] = "uid0003"
    pd.concat([gi, extra], ignore_index=True).to_parquet(tmp_path / "gene_index.parquet")
    cat = Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache")
    assert [d.uid for d in cat.genes.datasets_with("Alb", validation="pass")] == ["uid0003"]
    assert sorted(d.uid for d in cat.genes.datasets_with("Alb")) == ["uid0003", "uid0004"]


def test_refresh_passed_to_sidecar(tmp_path, monkeypatch):
    import sddb.genes as genes_mod

    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    cat = Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache", refresh=True)
    seen = []
    real = genes_mod.fetch_catalog

    def spy(url, **kw):
        seen.append(kw.get("refresh"))
        return real(url, **kw)

    monkeypatch.setattr(genes_mod, "fetch_catalog", spy)
    cat.genes  # noqa: B018
    assert seen == [True]


def test_where_expressed_is_datasets_with(tmp_path):
    cat = _cat(tmp_path)
    a = cat.genes.where_expressed("EPCAM").to_df()
    b = cat.genes.datasets_with("EPCAM").to_df()
    assert list(a["uid"]) == list(b["uid"])
    assert [d.uid for d in cat.genes.where_expressed("EPCAM", min_fraction=0.3)] == ["uid0001", "uid0002"]


def _stub_genes(tmp_path):
    """Gene index with nullable dtypes: bronze uid0004 expresses EPCAM; MALAT1 is legacy (NA feature_id)."""
    import pandas as pd

    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    rows = [
        ("EPCAM", "ENSG1", "uid0001", 5000.0, 0.4),
        ("EPCAM", "ENSG1", "uid0003", 3000.0, 0.3),
        ("EPCAM", "ENSG1", "uid0004", 100.0, 0.1),  # uid0004 is validation=fail
        ("CD3D", "ENSG2", "uid0001", 900.0, 0.5),
        ("MALAT1", None, "uid0002", 50.0, 0.9),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "feature_id", "uid", "total_counts", "fraction_obs_detected"])
    df = df.astype(
        {"symbol": "string", "feature_id": "string", "uid": "string", "total_counts": "Float64",
         "fraction_obs_detected": "Float64"}
    )
    assert df["feature_id"].isna().sum() == 1
    df.to_parquet(tmp_path / "gene_index.parquet")
    return Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache").genes


def _uids(c):
    return sorted(d.uid for d in c)


def test_symbol_and_ensembl_resolve_same(tmp_path):
    g = _stub_genes(tmp_path)
    assert _uids(g.where_expressed("EPCAM")) == _uids(g.where_expressed(" ensg1 ")) == ["uid0001", "uid0003", "uid0004"]
    assert g.resolve(["epcam", "nope"]) == {"epcam": {"uid0001", "uid0003", "uid0004"}, "nope": set()}


def test_and_vs_any(tmp_path):
    g = _stub_genes(tmp_path)
    assert _uids(g.where_expressed(["EPCAM", "CD3D"], mode="all")) == ["uid0001"]
    assert _uids(g.where_expressed(["EPCAM", "CD3D"], mode="any")) == ["uid0001", "uid0003", "uid0004"]
    assert _uids(g.where_expressed(["EPCAM", "NOPE"], mode="all")) == []
    assert _uids(g.where_expressed(["EPCAM"], mode="all", min_fraction=0.3)) == ["uid0001", "uid0003"]


def test_bronze_default_and_pass_filter(tmp_path):
    g = _stub_genes(tmp_path)
    df = g.where_expressed("EPCAM").to_df().set_index("uid")
    assert "uid0004" in df.index
    assert df.loc["uid0004", "validation_status"] == "fail"
    assert "tier" in df.columns
    assert _uids(g.where_expressed("EPCAM", validation="pass")) == ["uid0001", "uid0003"]


def test_legacy_symbol_only_no_crash(tmp_path):
    g = _stub_genes(tmp_path)
    assert _uids(g.where_expressed("MALAT1")) == ["uid0002"]
    assert _uids(g.where_expressed(["MALAT1", "EPCAM"], mode="any")) == ["uid0001", "uid0002", "uid0003", "uid0004"]


def test_backcompat_single_shape(tmp_path):
    g = _stub_genes(tmp_path)
    a, b = g.where_expressed("EPCAM").to_df(), g.datasets_with("EPCAM").to_df()
    assert list(a.columns) == list(b.columns) and list(a["uid"]) == list(b["uid"])
