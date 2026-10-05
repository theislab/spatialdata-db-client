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
