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
    assert [d.uid for d in cat.genes.datasets_with("Alb")] == ["uid0003"]
    assert sorted(d.uid for d in cat.genes.datasets_with("Alb", validation=None)) == ["uid0003", "uid0004"]


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
