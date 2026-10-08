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


# ---------------------------------------------------------------------------
# Cross-surface consistency guard (spec §3, shared canonical rules).
#
# The fixture below and the expected memberships in CANONICAL_MEMBERSHIP are the
# SHARED expectation that sub-project A's engine tests also encode
# (build_gene_entities / build_gene_search in the spatialdata-db repo). The web
# consumes A's compact artifact and the client queries gene_index.parquet directly;
# both must agree on gene -> dataset membership. This is a client-side guard: it
# does NOT import engine code. A change to the canonical rules (spec §3: Ensembl-keyed
# entities, symbol-keyed legacy entities, AND = intersection, bronze included by
# default and flagged, validation="pass" restricts) MUST break BOTH repos' tests.
#
# Shared fixture (nullable "string" dtypes so pd.NA is a real missing feature_id):
#   EPCAM / ENSG1      in {uid0001, uid0003}     Ensembl-keyed, several datasets
#   CD3D  / ENSG2      in {uid0001, uid0004}     uid0004 is BRONZE (validation "fail")
#   MALAT1 / <NA>      in {uid0002}              legacy symbol-only -> sym:malat1 entity
# ---------------------------------------------------------------------------

CANONICAL_MEMBERSHIP = {
    # symbol and Ensembl id resolve to the same entity -> same uid set
    "EPCAM": {"uid0001", "uid0003"},
    "ENSG1": {"uid0001", "uid0003"},
    # multi-gene AND = intersection; OR = union
    "AND(EPCAM,CD3D)": {"uid0001"},
    "OR(EPCAM,CD3D)": {"uid0001", "uid0003", "uid0004"},
    # bronze: present by default (flagged), excluded under validation="pass"
    "CD3D": {"uid0001", "uid0004"},
    "CD3D@pass": {"uid0001"},
    # legacy symbol-only gene (NA feature_id) resolves by its symbol
    "MALAT1": {"uid0002"},
}


def _canonical_gene_index():
    import pandas as pd

    rows = [
        ("EPCAM", "ENSG1", "uid0001", 5000.0, 0.4),
        ("EPCAM", "ENSG1", "uid0003", 3000.0, 0.3),
        ("CD3D", "ENSG2", "uid0001", 900.0, 0.5),
        ("CD3D", "ENSG2", "uid0004", 120.0, 0.2),  # bronze: uid0004 validation_status "fail"
        ("MALAT1", pd.NA, "uid0002", 50.0, 0.9),  # legacy symbol-only row
    ]
    df = pd.DataFrame(rows, columns=["symbol", "feature_id", "uid", "total_counts", "fraction_obs_detected"])
    return df.astype(
        {"symbol": "string", "feature_id": "string", "uid": "string", "total_counts": "Float64",
         "fraction_obs_detected": "Float64"}
    )


def test_cross_surface_canonical_membership(tmp_path):
    """Client membership equals the canonical expectation shared with A's engine tests (spec §3)."""
    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    df = _canonical_gene_index()
    assert df["feature_id"].isna().sum() == 1  # legacy row is a real pd.NA
    df.to_parquet(tmp_path / "gene_index.parquet")
    g = Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache").genes

    # symbol and Ensembl id -> same entity
    assert _uids(g.where_expressed("EPCAM")) == sorted(CANONICAL_MEMBERSHIP["EPCAM"])
    assert _uids(g.where_expressed("ENSG1")) == sorted(CANONICAL_MEMBERSHIP["ENSG1"])
    # multi-gene AND / OR
    assert _uids(g.where_expressed(["EPCAM", "CD3D"], mode="all")) == sorted(CANONICAL_MEMBERSHIP["AND(EPCAM,CD3D)"])
    assert _uids(g.where_expressed(["EPCAM", "CD3D"], mode="any")) == sorted(CANONICAL_MEMBERSHIP["OR(EPCAM,CD3D)"])
    # bronze: default includes and flags; validation="pass" excludes
    assert _uids(g.where_expressed("CD3D")) == sorted(CANONICAL_MEMBERSHIP["CD3D"])
    cd3d = g.where_expressed("CD3D").to_df().set_index("uid")
    assert cd3d.loc["uid0004", "validation_status"] == "fail"
    assert _uids(g.where_expressed("CD3D", validation="pass")) == sorted(CANONICAL_MEMBERSHIP["CD3D@pass"])
    # legacy symbol-only gene resolves by symbol
    assert _uids(g.where_expressed("MALAT1")) == sorted(CANONICAL_MEMBERSHIP["MALAT1"])
