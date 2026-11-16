import pandas as pd
import pytest
from conftest import IMAGING_GENES, SPOT_GENES, member, symbols
from primers.cohort import (
    NO_PAPER,
    NO_SYMBOLS,
    TENX_DEMO_PAPER,
    TENX_RERELEASE,
    WRONG_ORGANISM,
    hd_table,
    imaging_view_allowed,
    load_cohort,
    paper_for_members,
    smoke_spec,
    tenx_sections,
)


def _members(rows):
    return pd.DataFrame(rows, columns=["uid", "study_id", "collections", "doi"])


def test_paper_rule_order():
    """Own DOI, then GEO series, then collection, then study_id."""
    df = _members(
        [
            # own DOI wins
            ("u1", "geo:GSM1", ["geo-gse10"], "10.1/a"),
            # GEO series beats the other collection slug
            ("u2", "geo:GSM2", ["oliveira-2024", "geo-gse20"], None),
            # collection slug when there is no GEO series
            ("u3", "hest:X3", ["smith-2023-lung"], None),
            # study_id when there are no collections
            ("u4", "10x:demo4", None, None),
            ("u5", "10x:demo5", [], ""),
            # a study id and the matching collection slug are one paper
            ("u6", "10x:Demo5", None, None),
            ("u7", None, ["10x-demo5"], None),
            # neither: no paper
            ("u8", None, None, None),
        ]
    )
    out = paper_for_members(df, merge_10x_demos=False).set_index("uid")
    assert out.loc["u1", ["paper", "paper_source"]].tolist() == ["10.1/a", "doi"]
    assert out.loc["u2", ["paper", "paper_source"]].tolist() == ["geo-gse20", "geo_series"]
    assert out.loc["u3", ["paper", "paper_source"]].tolist() == ["smith-2023-lung", "collection"]
    assert out.loc["u4", ["paper", "paper_source"]].tolist() == ["10x-demo4", "study_id"]
    assert out.loc["u5", ["paper", "paper_source"]].tolist() == ["10x-demo5", "study_id"]
    assert out.loc["u6", "paper"] == out.loc["u7", "paper"] == "10x-demo5"
    assert out.loc["u7", "paper_source"] == "collection"
    assert pd.isna(out.loc["u8", "paper"])
    assert out.loc["u8", "paper_source"] == "none"
    assert (out["paper_inferred_from"] == "").all()
    merged = paper_for_members(df).set_index("uid")
    assert merged.loc[["u4", "u5", "u6", "u7"], "paper_source"].tolist() == ["10x_demo"] * 4
    assert merged.loc[["u1", "u2", "u3"], "paper"].tolist() == out.loc[["u1", "u2", "u3"], "paper"].tolist()


def test_paper_doi_propagates_within_series():
    """One DOI in a series or collection reaches every member and records its carrier."""
    df = _members(
        [
            ("a1", "geo:GSM1", ["geo-gse99"], None),
            ("a2", "geo:GSM2", ["geo-gse99"], "10.1/x"),
            ("a3", "geo:GSM3", ["geo-gse99", "some-collection"], None),
            # a different series stays its own paper
            ("b1", "geo:GSM4", ["geo-gse77"], None),
            # a collection group propagates too
            ("c1", "hest:C1", ["hest-colon"], "10.2/y"),
            ("c2", "hest:C2", ["hest-colon"], None),
        ]
    )
    out = paper_for_members(df).set_index("uid")
    assert set(out.loc[["a1", "a2", "a3"], "paper"]) == {"10.1/x"}
    assert out.loc["a2", "paper_source"] == "doi"
    assert out.loc[["a1", "a3"], "paper_source"].tolist() == ["doi_inferred", "doi_inferred"]
    assert out.loc[["a1", "a3"], "paper_inferred_from"].tolist() == ["a2", "a2"]
    assert out.loc["b1", ["paper", "paper_source"]].tolist() == ["geo-gse77", "geo_series"]
    assert out.loc["c2", ["paper", "paper_source", "paper_inferred_from"]].tolist() == ["10.2/y", "doi_inferred", "c1"]
    assert out["paper"].nunique() == 3


# load_cohort and the cohort table

SPEC = {"slug": "test", "query": {"organism": "human"}}


def _status(table):
    return table.set_index("uid")["status"].to_dict()


def test_drops_non_redistributable(make_catalog):
    """A member whose licence does not allow redistribution is dropped with that reason."""
    cat = make_catalog([member("a"), member("b"), member("c"), member("d", license="unknown")])
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert _status(table)["d"] == "dropped: licence not redistributable"
    assert set(adatas["spot"].obs["dataset_uid"]) == {"a", "b", "c"}


def test_dropped_member_recorded(make_catalog):
    """A store that fails to read is recorded as dropped, and the other members still load."""
    cat = make_catalog([member("a"), member("b"), member("c"), member("d")], broken={"b"})
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert _status(table)["b"] == "dropped: load error: KeyError"
    assert set(adatas["spot"].obs["dataset_uid"]) == {"a", "c", "d"}
    assert list(table.columns) == [
        "uid",
        "title",
        "technology",
        "family",
        "paper",
        "paper_source",
        "paper_inferred_from",
        "license",
        "license_noncommercial",
        "n_obs",
        "n_genes",
        "n_ensembl",
        "status",
    ]


def test_nc_licence_kept_and_flagged(make_catalog):
    """A noncommercial licence is kept, and the table and obs flag it."""
    cat = make_catalog([member("a"), member("b", license="CC-BY-NC-4.0"), member("c")])
    adatas, table = load_cohort(SPEC, catalog=cat)
    row = table.set_index("uid").loc["b"]
    assert row["status"] == "kept"
    assert bool(row["license_noncommercial"])
    obs = adatas["spot"].obs
    assert set(obs.loc[obs["dataset_uid"] == "b", "license"]) == {"CC-BY-NC-4.0"}
    assert set(obs.loc[obs["dataset_uid"] == "b", "license_noncommercial"].astype(str)) == {"True"}


def test_refuses_fewer_than_3_papers(make_catalog):
    """Fewer than 3 papers after the licence filter is refused with the count."""
    # three sections but two papers: a and b share a GEO series
    cat = make_catalog(
        [
            member("a", collections=["geo-gse1"]),
            member("b", collections=["geo-gse1"]),
            member("c"),
            member("d", license="unknown"),
        ]
    )
    with pytest.raises(ValueError, match="has 2 papers"):
        load_cohort(SPEC, catalog=cat)


def test_refuses_when_load_errors_leave_fewer_than_3_papers(make_catalog):
    """Load errors that leave fewer than 3 papers are refused too."""
    cat = make_catalog([member("a"), member("b"), member("c")], broken={"c"})
    with pytest.raises(ValueError, match="has 2 papers after loading"):
        load_cohort(SPEC, catalog=cat)


def test_families_loaded_separately(make_catalog):
    """Spot and imaging members give two AnnData objects, each with its own genes."""
    cat = make_catalog([member("a"), member("b"), member("x", "Xenium"), member("m", "Merscope")])
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert set(adatas) == {"spot", "imaging_cell"}
    assert list(adatas["spot"].var_names) == symbols(SPOT_GENES)
    assert list(adatas["imaging_cell"].var_names) == symbols(IMAGING_GENES)
    assert set(adatas["imaging_cell"].obs["dataset_uid"]) == {"x", "m"}
    assert table.set_index("uid")["family"].to_dict() == {
        "a": "spot",
        "b": "spot",
        "x": "imaging_cell",
        "m": "imaging_cell",
    }


def test_obs_columns_present(make_catalog):
    """Every obs column is present, and coordinates are filled."""
    cat = make_catalog([member("a", doi="10.1/a"), member("b"), member("c")])
    adatas, _ = load_cohort(SPEC, catalog=cat)
    obs = adatas["spot"].obs
    for col in (
        "dataset_uid",
        "paper",
        "paper_source",
        "technology",
        "tissue",
        "donor_id",
        "license",
        "license_noncommercial",
    ):
        assert col in obs.columns, col
    assert obs.loc[obs["dataset_uid"] == "a", "paper"].iloc[0] == "10.1/a"
    assert adatas["spot"].obsm["spatial"].shape == (adatas["spot"].n_obs, 2)


def test_include_and_exclude_uids(make_catalog):
    """include_uids adds members to the query and exclude_uids removes them."""
    cat = make_catalog([member("a"), member("b"), member("c"), member("d")])
    spec = {"slug": "t", "query": {"study_id": ["geo:a", "geo:b"]}, "include_uids": ["c", "d"], "exclude_uids": ["d"]}
    _, table = load_cohort(spec, catalog=cat)
    assert sorted(table["uid"]) == ["a", "b", "c"]


def test_10x_demos_count_as_one_paper(make_catalog):
    """All 10x demo releases count as one paper."""
    cat = make_catalog(
        [member("a", study_id="10x:demo-a"), member("b", study_id="10x:demo-b"), member("c"), member("d")]
    )
    _, table = load_cohort(SPEC, catalog=cat)
    t = table.set_index("uid")
    assert t.loc[["a", "b"], "paper"].tolist() == [TENX_DEMO_PAPER] * 2
    assert t.loc[["a", "b"], "paper_source"].tolist() == ["10x_demo"] * 2
    assert t["paper"].nunique() == 3
    cat = make_catalog([member("a", study_id="10x:demo-a"), member("b", study_id="10x:demo-b"), member("c")])
    with pytest.raises(ValueError, match="has 2 papers"):
        load_cohort(SPEC, catalog=cat)


def test_tenx_sections_drops_rereleases_and_merges_blocks():
    """10x re-releases of one section keep the newest release; serial sections of one block share a block."""
    keys = pd.Series(
        [
            "10x:heart-1-standard-1-1-0",
            "10x:heart-1-standard-1-0-0",
            "10x:brain-serial-section-2-anterior-1-standard",
            "10x:brain-serial-section-2-anterior-1-standard-1-1-0",
            "10x:brain-serial-section-1-anterior-1-standard-1-0-0",
            "geo:gse1",
            "geo:gse1",
        ],
        index=list("abcdefg"),
    )
    rerelease, block = tenx_sections(keys)
    assert rerelease.tolist() == [False, True, False, True, False, False, False]
    assert block["b"] == "10x:heart-1-standard"
    assert block["d"] == block["e"] == "10x:brain-serial"
    assert pd.isna(block["f"])


def test_10x_rerelease_dropped_and_block_in_obs(make_catalog):
    """A 10x re-release is dropped from the cohort; kept serial sections carry their block in obs."""
    cat = make_catalog(
        [
            member("a", study_id="10x:heart-1-standard-1-0-0"),
            member("a2", study_id="10x:heart-1-standard-1-1-0"),
            member("s1", study_id="10x:blk-section-1-1-standard-1-0-0"),
            member("s2", study_id="10x:blk-section-2-1-standard-1-0-0"),
            member("c"),
            member("d"),
        ]
    )
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert _status(table)["a"] == TENX_RERELEASE
    assert _status(table)["a2"] == "kept"
    obs = adatas["spot"].obs.drop_duplicates("dataset_uid").set_index("dataset_uid")["block"]
    assert obs["s1"] == obs["s2"] == "10x:blk"
    assert obs["c"] == "c"


def test_member_without_any_paper_key_dropped(make_catalog):
    """A member with no study_id, series or collection is dropped, never its own paper."""
    cat = make_catalog([member("a"), member("b"), member("c"), member("orphan", study_id="")])
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert _status(table)["orphan"] == NO_PAPER
    assert "orphan" not in set(adatas["spot"].obs["dataset_uid"])
    # three orphans never make three papers
    cat = make_catalog([member("a"), member("b")] + [member(f"o{i}", study_id="") for i in range(3)])
    with pytest.raises(ValueError, match="has 2 papers"):
        load_cohort(SPEC, catalog=cat)


def test_uid_keyed_member_grouped_by_series():
    """A member without a study_id is grouped by its GEO series."""
    df = _members(
        [
            ("s1", None, ["geo-gse5"], None),
            ("s2", "geo:GSM2", ["geo-gse5"], "10.5/z"),
            ("s3", None, ["geo-gse6"], None),
        ]
    )
    out = paper_for_members(df).set_index("uid")
    assert out.loc["s1", ["paper", "paper_source", "paper_inferred_from"]].tolist() == ["10.5/z", "doi_inferred", "s2"]
    assert out.loc["s3", ["paper", "paper_source"]].tolist() == ["geo-gse6", "geo_series"]


def test_imaging_view_allowed():
    """The imaging view needs at least 2 papers or 3 kept sections."""

    def table(rows):
        return pd.DataFrame(rows, columns=["uid", "family", "paper", "status"])

    one_paper_two = table(
        [("x1", "imaging_cell", "p1", "kept"), ("x2", "imaging_cell", "p1", "kept"), ("v", "spot", "p2", "kept")]
    )
    assert imaging_view_allowed(one_paper_two) == (False, "1 paper, 2 sections")
    three = table([(f"x{i}", "imaging_cell", "p1", "kept") for i in range(3)])
    assert imaging_view_allowed(three) == (True, "1 paper, 3 sections")
    two_papers = table([("x1", "imaging_cell", "p1", "kept"), ("x2", "imaging_cell", "p2", "kept")])
    assert imaging_view_allowed(two_papers) == (True, "2 papers, 2 sections")
    dropped = table(
        [("x1", "imaging_cell", "p1", "kept"), ("x2", "imaging_cell", "p2", "dropped: load error: KeyError")]
    )
    assert imaging_view_allowed(dropped) == (False, "1 paper, 1 section")


def test_member_without_ensembl_ids_kept(make_catalog):
    """A table keyed by symbols with no Ensembl ids (Allen MERSCOPE style) is kept and joins its family."""
    cat = make_catalog([member("a"), member("b"), member("x", "Xenium"), member("m", "Merscope", ensembl=False)])
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert _status(table)["m"] == "kept"
    imaging = adatas["imaging_cell"]
    assert list(imaging.var_names) == symbols(IMAGING_GENES)
    assert (imaging[imaging.obs["dataset_uid"] == "m"].X == 1).all()


def test_member_genes_and_ensembl_coverage_recorded(make_catalog):
    """Each family records the genes each member measured, and the table gives n_genes and n_ensembl."""
    cat = make_catalog([member("a"), member("b"), member("x", "Xenium"), member("m", "Merscope", ensembl=False)])
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert adatas["imaging_cell"].uns["member_genes"] == {
        "x": symbols(IMAGING_GENES),
        "m": symbols(IMAGING_GENES),
    }
    assert set(adatas["spot"].uns["member_genes"]) == {"a", "b"}
    t = table.set_index("uid")
    assert t.loc["a", "n_genes"] == len(SPOT_GENES)
    assert t.loc["a", "n_ensembl"] == len(SPOT_GENES)
    assert t.loc["m", "n_genes"] == len(IMAGING_GENES)
    assert t.loc["m", "n_ensembl"] == 0


def test_member_without_symbols_dropped_with_reason(make_catalog):
    """A table with no gene symbols is dropped with that reason, and the others still load."""
    cat = make_catalog([member("a"), member("b"), member("c"), member("d", keys=False)])
    adatas, table = load_cohort(SPEC, catalog=cat)
    assert _status(table)["d"] == NO_SYMBOLS
    assert set(adatas["spot"].obs["dataset_uid"]) == {"a", "b", "c"}


def _var_only(names, **columns):
    import anndata as ad

    return ad.AnnData(var=pd.DataFrame(columns, index=names))


def test_restrict_to_organism_prefixed_tables():
    """Genome prefixes: keep the organism's genome and strip it; a pathogen genome is dropped."""
    from primers.cohort import restrict_to_organism

    xeno = _var_only([f"hg38_H{i}" for i in range(1200)] + [f"mm10_M{i}" for i in range(1100)])
    out = restrict_to_organism(xeno, "human")
    assert out.n_vars == 1200
    assert out.var_names[0] == "H0"
    host = _var_only([f"mm39_M{i}" for i in range(1500)] + [f"ANKA_P{i}" for i in range(1000)])
    assert restrict_to_organism(host, "mouse").var_names.tolist() == [f"M{i}" for i in range(1500)]
    single = _var_only([f"GRCh38__G{i}" for i in range(1000)])
    assert restrict_to_organism(single, "human").var_names[-1] == "G999"


def test_restrict_to_organism_by_ensembl_ids():
    """Without prefixes the Ensembl ids decide; a few underscores in names are not a genome prefix."""
    from primers.cohort import WrongOrganismError, restrict_to_organism

    ids = [f"ENSG{i:011d}" for i in range(3)] + [f"ENSMUSG{i:011d}" for i in range(2)]
    dual = _var_only(["KRT14", "KRT5", "MT_CO1", "Krt14", "Krt5"], gene_ids=ids)
    assert restrict_to_organism(dual, "human").var_names.tolist() == ["KRT14", "KRT5", "MT_CO1"]
    assert restrict_to_organism(dual, "mouse").var_names.tolist() == ["Krt14", "Krt5"]
    human_only = _var_only(["ALB", "TTR"], gene_ids=ids[:2])
    with pytest.raises(WrongOrganismError, match="none of mouse"):
        restrict_to_organism(human_only, "mouse")
    assert restrict_to_organism(human_only, "zebrafish") is human_only
    no_ids = _var_only(["Alb", "Ttr"])
    assert restrict_to_organism(no_ids, "mouse") is no_ids


def test_member_with_another_organisms_genes_dropped(make_catalog):
    """A member labelled mouse whose table holds human Ensembl ids is dropped with a reason."""
    cat = make_catalog([member(u) for u in ("a", "b", "c")] + [member("m", organism="mouse")])
    adatas, table = load_cohort({"slug": "s", "query": {}}, catalog=cat, max_workers=1)
    status = table.set_index("uid")["status"]
    assert status["m"] == WRONG_ORGANISM
    assert (status.drop("m") == "kept").all()


def test_symbols_like_ensa_are_not_ensembl_ids():
    """A symbol column holding ENSA is not read as Ensembl ids; a later column with real ids is."""
    from primers.cohort import restrict_to_organism

    dual = _var_only(
        ["ENSA", "KRT5", "Krt5"],
        gene_id=["ENSA", "KRT5", "Krt5"],  # checked before original_var_name
        original_var_name=["ENSG00000143420", "ENSG00000186081", "ENSMUSG00000061527"],
    )
    assert restrict_to_organism(dual, "human").var_names.tolist() == ["ENSA", "KRT5"]


def test_host_without_prefix_and_pathogen_gene_ids():
    """Unprefixed host genes are kept when the pathogen genes share a gene-id prefix (PBANKA_...)."""
    from primers.cohort import WrongOrganismError, restrict_to_organism

    host = [f"Gene{i}" for i in range(1200)]
    pathogen = [f"PBANKA_{i:07d}" for i in range(1100)]
    ids = [f"ENSMUSG{i:011d}" for i in range(1200)] + pathogen
    table = _var_only(host + pathogen, gene_ids=ids)
    assert restrict_to_organism(table, "mouse").var_names.tolist() == host
    with pytest.raises(WrongOrganismError):
        restrict_to_organism(table, "human")
    # a lone transgene without an Ensembl id is kept
    gfp = _var_only(["Alb", "Ttr", "eGFP"], gene_ids=["ENSMUSG00000029368", "ENSMUSG00000061808", "eGFP"])
    assert restrict_to_organism(gfp, "mouse").var_names.tolist() == ["Alb", "Ttr", "eGFP"]


def test_hd_table_falls_back_to_8um():
    """A Visium HD store without the 16 um table uses the 8 um one."""
    assert hd_table({"tables/square_016um", "tables/square_008um"}) == "square_016um"
    assert hd_table(["tables/square_008um", "shapes/x"]) == "square_008um"


def test_aggregate_bins_sums_2x2_blocks():
    """Four 8 um bins of one 2 x 2 block become one bin with summed counts and the mean position."""
    import anndata as ad
    import numpy as np
    from primers.cohort import aggregate_bins

    obs = pd.DataFrame(
        {"array_row": [0, 0, 1, 1, 2], "array_col": [0, 1, 0, 1, 0], "dataset_uid": "u"}, index=list("abcde")
    )
    X = np.array([[1, 0], [2, 1], [3, 0], [4, 2], [5, 5]], dtype="float32")
    a = ad.AnnData(
        X, obs=obs, var=pd.DataFrame(index=["G1", "G2"]), obsm={"spatial": np.arange(10, dtype=float).reshape(5, 2)}
    )
    out = aggregate_bins(a)
    assert out.n_obs == 2
    np.testing.assert_array_equal(out.X.toarray(), [[10, 3], [5, 5]])
    np.testing.assert_allclose(out.obsm["spatial"][0], [3.0, 4.0])
    assert out.obs[["array_row", "array_col"]].values.tolist() == [[0, 0], [1, 0]]
    assert out.obs_names.is_unique


def test_smoke_spec_keeps_smallest_member_per_paper_alternating_families(make_catalog):
    """``smoke_spec`` keeps the smallest kept member of each of n papers, taken in turn from each family."""
    cat = make_catalog(
        [
            member("v1", n=5),
            member("v2", n=3, study_id="geo:v1"),
            member("x1", "Xenium", n=4),
            member("v3", n=6),
            member("x2", "Xenium", n=7),
        ]
    )
    spec = {"slug": "s", "query": {}, "exclude_uids": ["zz"]}
    out = smoke_spec(spec, cat, 3)
    assert out["slug"] == "s-smoke"
    assert set(out["exclude_uids"]) == {"zz", "v1", "v3"}  # kept: x1 and v2 (round 1), then x2


def test_genome_prefix_with_three_underscores():
    """GEO xenograft tables may use mm10___ (three underscores); it is still a genome prefix."""
    from primers.cohort import restrict_to_organism

    table = _var_only([f"GRCh38_H{i}" for i in range(1100)] + [f"mm10___M{i}" for i in range(1200)])
    assert restrict_to_organism(table, "mouse").var_names.tolist() == [f"M{i}" for i in range(1200)]
