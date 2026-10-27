from __future__ import annotations

import pandas as pd
import pytest
from tests._fixtures import write_fixture_catalog, write_fixture_feature_index, write_fixture_gene_index

from sddb import Catalog
from sddb.mcp import tools


def _cat(tmp_path):
    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    write_fixture_feature_index(tmp_path / "feature_index.parquet")
    return Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache")


def test_resolve_protein_target(tmp_path):
    hits = _cat(tmp_path).features.resolve("anti-epcam")
    assert list(hits["feature_type"]) == ["protein"]
    assert hits.iloc[0]["uids"] == ["uid0003"]
    by_ref = _cat(tmp_path).features.resolve("P16422", feature_type="protein")
    assert list(by_ref["raw_label"]) == ["anti-EpCAM"]


def test_resolve_metabolite_by_hmdb_and_name(tmp_path):
    f = _cat(tmp_path).features
    assert list(f.resolve("HMDB0000122")["raw_label"]) == ["Glucose"]
    assert list(f.resolve("glucose", feature_type="metabolite")["modality"]) == ["msi"]
    assert f.resolve("glucose", feature_type="gene").empty


def test_by_gene_cross_modal(tmp_path):
    cat = _cat(tmp_path)
    assert sorted(d.uid for d in cat.features.by_gene("EPCAM")) == ["uid0001", "uid0002", "uid0003", "uid0005"]
    assert sorted(d.uid for d in cat.features.by_gene("ENSG00000119888")) == ["uid0001", "uid0002", "uid0003", "uid0005"]
    assert len(cat.features.by_gene("NOPE")) == 0
    assert [d.uid for d in cat.features.by_gene("KRT8")] == ["uid0001"]


def test_where_measured_and_ranked(tmp_path):
    f = _cat(tmp_path).features
    assert sorted(d.uid for d in f.where_measured("EPCAM", feature_type="gene")) == ["uid0001", "uid0002"]
    assert [d.uid for d in f.where_measured(["EPCAM", "KRT8"], mode="all")] == ["uid0001"]
    assert sorted(d.uid for d in f.where_measured(["Glucose", "KRT8"], mode="any")) == ["uid0001", "uid0004"]
    assert [d.uid for d in f.where_measured("EPCAM", modality="atac")] == ["uid0005"]
    assert list(f.ranked("EPCAM", feature_type="gene")["uid"]) == ["uid0001", "uid0002"]
    with pytest.raises(ValueError):
        f.where_measured("EPCAM", mode="xor")


def test_gene_index_backcompat_view(tmp_path):
    cat = _cat(tmp_path)
    assert sorted(d.uid for d in cat.genes.datasets_with("epcam")) == ["uid0001", "uid0002", "uid0005"]
    assert cat.genes.resolve("KRT8") == {"KRT8": {"uid0001"}}


def test_features_tool_typed_hits(tmp_path):
    _cat(tmp_path)
    url = (tmp_path / "catalog.parquet").as_uri()
    rows = tools.features_tool(["Glucose"], modality="msi", catalog_url=url)
    assert [r["uid"] for r in rows] == ["uid0004"]
    assert rows[0]["features"] == [{"raw_label": "Glucose", "feature_type": "metabolite", "modality": "msi"}]
    prot = tools.features_tool("anti-EpCAM", feature_type="protein", catalog_url=url)
    assert prot[0]["features"][0]["feature_type"] == "protein"


def test_suggest_tolerates_list_valued_facet(tmp_path):
    from sddb.cli import _suggest

    cat = _cat(tmp_path)
    df = cat.to_df()
    df["modalities"] = [["rna", "protein"], ["rna"], None, ["msi"], ["rna"]]
    cat._df = df
    _suggest(cat, {"modalities": "rnaa"})  # must not raise on list cells


def test_group_counts_explodes_list_column():
    from sddb.cohort import group_counts

    df = pd.DataFrame({"modalities": [["rna", "protein"], ["rna"], None, []]})
    out = dict(zip(group_counts(df, "modalities")["modalities"], group_counts(df, "modalities")["n"], strict=True))
    assert out == {"rna": 2, "protein": 1, "unknown": 2}


def test_resolve_tolerates_null_list_cells(tmp_path):
    from tests._fixtures import make_fixture_feature_index

    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    df = make_fixture_feature_index()
    df["gene_xref"] = df["gene_xref"].astype(object)
    df.at[5, "gene_xref"] = None
    df.at[5, "ref_ids"] = None
    df.at[5, "ref_id_types"] = None
    df.to_parquet(tmp_path / "feature_index.parquet")
    cat = Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache")
    hit = cat.features.resolve("glucose")
    assert hit.iloc[0]["gene_xref"] == [] and hit.iloc[0]["ref_ids"] == []
    assert hit.iloc[0]["uids"] == ["uid0004"]
