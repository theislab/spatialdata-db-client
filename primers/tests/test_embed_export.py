import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from primers.embed import _host, embed_views, joint_gates, mixing_stats
from primers.export import equal_share_subsample, export_bundle, mixing_by_view

RNG = np.random.default_rng(0)
COMPARTMENTS = {"epithelial": ["EPCAM"], "immune": ["PTPRC"]}
MARKERS = {"compartments": COMPARTMENTS}


def _family(sections, genes, tech, n=60, counts=50.0):
    """One family's AnnData: ``sections`` datasets of ``n`` observations, Poisson counts around ``counts``."""
    parts = []
    for s in sections:
        X = RNG.poisson(counts / len(genes), (n, len(genes))).astype("float32")
        obs = pd.DataFrame(
            {"dataset_uid": s, "technology": tech, "paper": f"paper-{s}", "tissue": "colon"},
            index=[f"{s}-{i}" for i in range(n)],
        )
        parts.append(ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes)))
    out = ad.concat(parts)
    out.obs["total_counts"] = np.asarray(out.X.sum(1)).ravel()
    out.obs["n_genes"] = np.asarray((out.X > 0).sum(1)).ravel()
    out.uns["member_genes"] = {s: list(genes) for s in sections}
    return out


def _members(parts, n=60, counts=50.0):
    """A family joined like ``load_cohort`` does: ``parts`` is (uid, technology, genes); missing genes are 0."""
    fams = [_family([uid], genes, tech, n=n, counts=counts) for uid, tech, genes in parts]
    out = ad.concat(fams, join="outer", fill_value=0)
    out.uns["member_genes"] = {uid: list(genes) for uid, _, genes in parts}
    return out


SHARED = ["EPCAM", "PTPRC"] + [f"S{i}" for i in range(248)]
SPOT = SHARED + [f"V{i}" for i in range(100)]
IMAGING = SHARED + [f"X{i}" for i in range(20)]


def test_subsample_equal_share_cap_50k():
    """At most the cap in total, and an equal share per dataset when all are large."""
    uids = pd.Series(np.repeat(["a", "b", "c", "d"], 30_000))
    idx = equal_share_subsample(uids, cap=50_000, seed=0)
    assert len(idx) == 50_000
    assert uids.iloc[idx].value_counts().tolist() == [12_500] * 4


def test_small_dataset_kept_whole():
    """A dataset smaller than its share is kept whole and its unused share goes to the others."""
    uids = pd.Series(["small"] * 100 + ["big1"] * 40_000 + ["big2"] * 40_000)
    counts = uids.iloc[equal_share_subsample(uids, cap=50_000, seed=0)].value_counts()
    assert counts["small"] == 100
    assert counts["big1"] == counts["big2"] == 24_950


def test_single_section_skips_harmony():
    """A view with one section is embedded uncorrected, with a note."""
    views = embed_views({"imaging_cell": _family(["x1"], IMAGING, "Xenium")}, MARKERS, floor=0, use_gpu=False)
    v = views["imaging_cell"]
    assert v.uns["primer"]["corrected"] is False
    assert "one section" in v.uns["primer"]["note"]
    assert "X_umap" in v.obsm
    assert "X_umap_harmony" not in v.obsm


def test_tiny_panel_member_does_not_erase_its_technology():
    """A member whose panel shares few genes with the rest is dropped, and its technology keeps a view."""
    panel = [f"M{i}" for i in range(120)]
    odd = panel[:3] + [f"Q{i}" for i in range(80)]  # shares 3 genes with the others
    fam = _members(
        [("m1", "Merscope", panel), ("m2", "Merscope", panel), ("m3", "Merscope", panel), ("odd", "Merscope", odd)]
    )
    views = embed_views({"imaging_cell": fam}, MARKERS, floor=0, use_gpu=False)
    (v,) = views.values()
    assert sorted(v.obs["dataset_uid"].astype(str).unique()) == ["m1", "m2", "m3"]
    assert v.uns["primer"]["genes"]["measured"] == 120
    assert v.uns["primer"]["members_dropped_panel"] == {
        "odd": "dropped: panel shares 3 genes with the rest of its technology"
    }


def test_panel_drops_are_greedy_and_stop_at_the_floor():
    """Members are dropped one at a time, the worst panel first, only until the technology shares 50 genes."""
    base = [f"G{i}" for i in range(100)]
    fam = _members(
        [
            ("a", "Merscope", base),
            ("b", "Merscope", base),
            ("c", "Merscope", base[:40] + [f"C{i}" for i in range(60)]),  # 40 shared: drops first
            ("d", "Merscope", base[:70] + [f"D{i}" for i in range(30)]),  # 70 shared: kept once c is gone
        ]
    )
    views = embed_views({"imaging_cell": fam}, MARKERS, floor=0, use_gpu=False)
    (v,) = views.values()
    assert list(v.uns["primer"]["members_dropped_panel"]) == ["c"]
    assert v.uns["primer"]["genes"]["measured"] == 70


def _panel_view(parts, tech="Xenium"):
    (v,) = embed_views(
        {"imaging_cell": _members([(u, tech, sorted(g)) for u, g in parts])}, MARKERS, floor=0, use_gpu=False
    ).values()
    return v


def test_big_panel_kept_at_a_cost_when_the_view_keeps_half_its_genes():
    """Kidney: a 5K panel sharing 129 of the 224 genes stays (58%), tiny 50-gene panels go, and the cost is logged."""
    shared = [f"G{i}" for i in range(224)]
    tiny = shared[:10] + [f"T{i}" for i in range(40)]
    big = shared[3:132] + [f"T{i}" for i in range(25)]  # 129 in common with the rest, 7 with the rest and tiny
    main = [(f"p{i}", shared) for i in range(1, 5)]  # the anchor: most sections, as the 13 in kidney
    v = _panel_view([*main, ("big", big)] + [(f"t{i}", tiny) for i in range(1, 4)])
    dropped = v.uns["primer"]["members_dropped_panel"]
    assert sorted(dropped) == ["t1", "t2", "t3"]
    assert set(dropped.values()) == {"dropped: panel shares 7 genes with the rest of its technology"}
    assert "big" in set(v.obs["dataset_uid"].astype(str))
    assert v.uns["primer"]["genes"]["measured"] == 129
    assert v.uns["primer"]["panel_costs"] == [
        {
            "technology": "Xenium",
            "uids": ["big"],
            "genes_without": 224,
            "genes_with": 129,
            "kept": True,
            "reason": "kept: shares 129 of 224",
        }
    ]


def test_panel_that_halves_the_view_is_dropped_with_its_cost():
    """Merscope: GEO sections sharing 61 of Allen's 500 genes go, so the view keeps 500 genes.

    Allen has the most sections, so it is the anchor even where the GEO sections hold more cells.
    """
    allen = {f"A{i}" for i in range(500)}
    geo = {f"A{i}" for i in range(61)} | {f"G{i}" for i in range(435)}
    tiny = {f"A{i}" for i in range(4)} | {f"G{i}" for i in range(4)} | {f"T{i}" for i in range(400)}
    parts = [(f"allen{i}", allen) for i in range(4)] + [(f"geo{i}", geo) for i in range(3)]
    v = _panel_view(parts + [("tiny1", tiny), ("tiny2", tiny)], tech="Merscope")
    dropped = v.uns["primer"]["members_dropped_panel"]
    assert sorted(v.obs["dataset_uid"].astype(str).unique()) == ["allen0", "allen1", "allen2", "allen3"]
    assert v.uns["primer"]["genes"]["measured"] == 500
    assert {dropped[f"geo{i}"] for i in range(3)} == {"dropped: panel would reduce 500 shared genes to 61"}
    assert dropped["tiny1"] == "dropped: panel shares 4 genes with the rest of its technology"
    (cost,) = v.uns["primer"]["panel_costs"]
    assert cost["kept"] is False
    assert (cost["genes_without"], cost["genes_with"], sorted(cost["uids"])) == (500, 61, ["geo0", "geo1", "geo2"])


def test_panel_dropped_member_marked_in_the_cohort_table(tmp_path):
    """The export marks a member dropped for its panel in the cohort table and the manifest."""
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    views = embed_views(fams, MARKERS, floor=10, use_gpu=False)
    reason = "dropped: panel shares 3 genes with the rest of its technology"
    for v in views.values():
        v.uns["primer"]["members_dropped_panel"] = {"x2": reason}
    m = json.loads((_bundle(tmp_path, views=views) / "manifest.json").read_text())
    assert m["members_dropped"]["x2"] == reason
    assert "x2" not in m["members_kept"]
    assert "members_dropped_panel" not in m["views"]["spot"]


def test_joint_view_only_when_gates_pass():
    """The joint view appears when all gates pass, and not when gate 1 fails (fewer than 200 shared genes)."""
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    views = embed_views(fams, MARKERS, floor=10, use_gpu=False)
    assert set(views) == {"spot", "imaging_cell", "joint"}
    assert views["joint"].n_vars == len(SHARED)
    assert views["joint"].uns["primer"]["corrected"] is True
    small = {
        "spot": fams["spot"][:, SPOT[:150] + SPOT[250:]],
        "imaging_cell": fams["imaging_cell"][:, IMAGING[:150] + IMAGING[250:]],
    }
    gates = joint_gates(small, MARKERS, floor=10)
    assert gates["gate1"]["passed"] is False
    assert "joint" not in embed_views(small, MARKERS, floor=10, use_gpu=False)


def test_failed_gate_excludes_technology_with_reason():
    """A technology that loses more than half its observations at the floor on the shared genes is left out."""
    fams = {
        "spot": _family(["v1", "v2"], SPOT, "Visium", counts=400.0),
        "imaging_cell": ad.concat(
            [
                _family(["x1", "x2"], IMAGING, "Xenium", counts=400.0),
                _family(["m1", "m2"], IMAGING, "Merscope", counts=5.0),
            ]
        ),
    }
    gates = joint_gates(fams, MARKERS, floor=50)
    assert gates["gate3"]["excluded"] == {"Merscope": gates["gate3"]["excluded"]["Merscope"]}
    assert "loses" in gates["gate3"]["excluded"]["Merscope"]
    joint = embed_views(fams, MARKERS, floor=50, use_gpu=False)["joint"]
    assert "Merscope" not in set(joint.obs["technology"])
    assert joint.uns["primer"]["excluded"]["Merscope"]


def test_mixing_stats_bounds():
    """Both statistics lie in [0, 1], with their random-mixing reference and a plain definition."""
    v = embed_views({"imaging_cell": _family(["x1", "x2", "x3"], IMAGING, "Xenium")}, MARKERS, floor=0, use_gpu=False)[
        "imaging_cell"
    ]
    for basis in ("X_pca", "X_pca_harmony"):
        s = mixing_stats(v, "dataset_uid", basis)
        for name in ("entropy", "other_share"):
            assert 0.0 <= s[name]["mean"] <= 1.0
            assert all(0.0 <= x <= 1.0 for x in s[name]["per_batch"].values())
            assert 0.0 <= s[name]["random"] <= 1.0
            assert s[name]["definition"]
        assert "not the same as" in s["caveat"]


def _bundle(tmp_path, views=None, cohort=None, qc=None, mixing=None, **kwargs):
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    views = views or embed_views(fams, MARKERS, floor=10, use_gpu=False)
    default_cohort = pd.DataFrame(
        {
            "uid": ["v1", "v2", "x1", "x2", "z9"],
            "paper": ["p1", "p2", "p3", "p4", "p5"],
            "status": ["kept"] * 4 + ["dropped: licence not redistributable"],
            "n_genes": pd.array([350, 350, 270, 270, None], dtype="Int64"),
            "n_ensembl": pd.array([350, 350, 0, 0, None], dtype="Int64"),
        }
    )
    default_qc = pd.DataFrame(
        {
            "dataset_uid": ["v1", "v2", "x1", "x2"],
            "n_before": [60] * 4,
            "n_after": [60] * 4,
            "flagged": [False] * 4,
            "status": ["kept"] * 4,
        }
    )
    cohort = default_cohort if cohort is None else cohort
    qc = default_qc if qc is None else qc
    spec = {"slug": "test-cohort", "title": "Test cohort", "query": {"tissue": "colon"}}
    return export_bundle(
        views,
        cohort,
        qc,
        spec,
        tmp_path,
        release="2026-10-28",
        runtime_s=1.5,
        mixing=mixing,
        primers_ref="abc123",
        **kwargs,
    )


def test_manifest_fields_present(tmp_path):
    """The manifest carries every field of the bundle contract."""
    out = _bundle(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    for key in (
        "slug",
        "title",
        "query",
        "release",
        "members_kept",
        "members_dropped",
        "n_datasets",
        "n_papers",
        "technologies",
        "n_cells_before",
        "n_cells_after",
        "n_flagged",
        "versions",
        "seeds",
        "hardware",
        "runtime_s",
        "path",
        "views",
        "gates",
        "mixing",
        "paper_inference",
    ):
        assert key in m, key
    assert m["members_dropped"] == {"z9": "dropped: licence not redistributable"}
    assert set(m["views"]) == {"spot", "imaging_cell", "joint"}
    assert m["views"]["joint"]["corrected"] is True
    umap = pd.read_parquet(out / "umap.parquet")
    assert set(umap["view"]) == {"spot", "imaging_cell", "joint"}
    assert {
        "x",
        "y",
        "x_harmony",
        "y_harmony",
        "dataset_uid",
        "paper",
        "technology",
        "tissue",
        "total_counts",
        "n_genes",
    } <= set(umap.columns)


def test_h5ad_has_no_X(tmp_path):
    """embedding.h5ad holds obs and float16 embeddings only; each observation appears once."""
    out = _bundle(tmp_path)
    e = ad.read_h5ad(out / "embedding.h5ad")
    assert e.n_vars == 0
    assert e.obsm["X_pca"].dtype == np.float16
    assert e.obsm["X_pca"].shape[1] == 30
    assert e.n_obs == 240
    assert e.obs_names.is_unique
    assert "X_umap_joint" in e.obsm


def test_unexpressed_genes_left_out():
    """Genes with no counts in a view are left out before PCA."""
    fam = _family(["x1", "x2"], IMAGING, "Xenium")
    fam.X[:, -1] = 0
    v = embed_views({"imaging_cell": fam}, MARKERS, floor=0, use_gpu=False)["imaging_cell"]
    assert IMAGING[-1] not in v.var_names
    assert v.uns["primer"]["genes"]["used"] == len(IMAGING) - 1


def test_views_keep_log_normalised_marker_expression():
    """obsm["marker_expr"] holds log1p(counts / size factor) of every measured marker and fallback gene.

    Genes match without regard to case, a measured but unexpressed marker is a column of zeros, and a
    marker the view did not measure is left out.
    """
    genes = [*IMAGING, "clca1"]
    fam = _family(["x1", "x2"], genes, "Xenium")
    fam.X[:, genes.index("S0")] = 0
    markers = {
        "compartments": {"epithelial": ["EPCAM", "S0"], "immune": ["PTPRC", "CD3E"], "goblet": ["MUC2"]},
        "fallback": {"Xenium-1-abc": {"goblet": {"genes": ["CLCA1"], "note": "goblet gene"}}},
    }
    v = embed_views({"imaging_cell": fam}, markers, floor=0, use_gpu=False)["imaging_cell"]
    expr = v.obsm["marker_expr"]
    assert isinstance(expr, pd.DataFrame)
    assert list(expr.index) == list(v.obs_names)
    assert sorted(expr.columns) == ["EPCAM", "PTPRC", "S0", "clca1"]
    raw = fam[v.obs_names, ["EPCAM", "PTPRC"]].X
    want = np.log1p(raw / v.obs["size_factor"].to_numpy()[:, None])
    np.testing.assert_allclose(expr[["EPCAM", "PTPRC"]].to_numpy(), want, rtol=1e-5)
    assert (expr["S0"] == 0).all()


# gene sets


def test_view_uses_genes_measured_by_every_member():
    """Members with 300 and 250 genes that share 220 give a view on the 220, not the outer join."""
    shared = ["EPCAM", "PTPRC"] + [f"S{i}" for i in range(218)]
    fam = _members(
        [
            ("x1", "Xenium", shared + [f"A{i}" for i in range(80)]),
            ("x2", "Xenium", shared + [f"B{i}" for i in range(30)]),
        ]
    )
    v = embed_views({"imaging_cell": fam}, MARKERS, floor=0, use_gpu=False)["imaging_cell"]
    assert sorted(v.var_names) == sorted(shared)
    assert v.uns["primer"]["genes"]["measured"] == 220


def test_family_under_200_shared_genes_splits_per_technology():
    """A family whose members share under 200 genes gets one view per technology, with the panels in the note.

    A technology whose own intersection has under 50 genes gets no view, and the reason is recorded.
    """
    common = [f"S{i}" for i in range(150)]
    fam = _members(
        [
            ("x1", "Xenium", common + [f"X{i}" for i in range(100)]),
            ("x2", "Xenium", common + [f"X{i}" for i in range(100)]),
            ("m1", "Merscope", common + [f"M{i}" for i in range(150)]),
            ("m2", "Merscope", common + [f"M{i}" for i in range(150)]),
            ("f1", "MERFISH", common[:40]),
        ]
    )
    views = embed_views({"imaging_cell": fam}, MARKERS, floor=0, use_gpu=False)
    assert set(views) == {"imaging_cell:Xenium", "imaging_cell:Merscope"}
    note = "MERFISH 40 ∩ Xenium 250 ∩ Merscope 300 = 40 shared genes (under 200), so one view per technology"
    for name, n in (("imaging_cell:Xenium", 250), ("imaging_cell:Merscope", 300)):
        p = views[name].uns["primer"]
        assert p["note"] == note
        assert p["family"] == "imaging_cell"
        assert p["genes"]["measured"] == n
        assert set(views[name].obs["technology"]) == {name.split(":")[1]}
        assert "under 50" in p["views_skipped"]["imaging_cell:MERFISH"]


def test_hvg_caps_at_2000_after_normalising_on_all_genes():
    """A 2,500-gene view keeps 2,000 HVGs; the size factors use the library size over all 2,500 genes."""
    genes = ["EPCAM", "PTPRC"] + [f"G{i}" for i in range(2498)]
    fam = _family(["v1", "v2"], genes, "Visium", counts=3000.0)
    v = embed_views({"spot": fam}, MARKERS, floor=0, use_gpu=False)["spot"]
    assert v.n_vars == 2000
    assert v.uns["primer"]["genes"] == {
        "measured": 2500,
        "used": 2000,
        "rule": v.uns["primer"]["genes"]["rule"],
        "size_factors": "all measured genes",
    }
    raw = np.asarray(fam.X.sum(1)).ravel()
    np.testing.assert_allclose(v.obs["size_factor"].to_numpy(), raw / np.median(raw), rtol=1e-5)
    assert v.X.nnz == 0  # views carry no expression


def test_gate2_counts_fallback_genes_and_fails_without_a_common_compartment():
    """A compartment is expected only when every family can measure it (marker or fallback gene).

    With no compartment measurable by every family, gate 2 fails and there is no joint view.
    """
    spot = _family(["v1", "v2"], ["EPCAM"] + [f"S{i}" for i in range(249)], "Visium")
    imaging = _family(["x1", "x2"], ["PTPRC"] + [f"S{i}" for i in range(249)], "Xenium")
    fams = {"spot": spot, "imaging_cell": imaging}
    gates = joint_gates(fams, MARKERS, floor=0)
    assert gates["gate2"]["passed"] is False
    assert gates["gate2"]["expected"] == []
    assert gates["reason"] == "no compartment measurable by every family"
    assert "joint" not in embed_views(fams, MARKERS, floor=0, use_gpu=False)
    with_fallback = {
        "compartments": COMPARTMENTS,
        "fallback": {
            "Xenium-250-x": {"epithelial": {"genes": ["S3"], "note": "n"}},
            "Visium-250-v": {"immune": {"not_measured": True, "note": "n"}},
        },
    }
    gates = joint_gates(fams, with_fallback, floor=0)
    assert gates["gate2"]["expected"] == ["epithelial"]
    assert gates["gate2"]["passed"] is True


def test_joint_gates_record_both_gene_counts():
    """Gate 3 runs on the genes of all members; gates 1 and 2 on those of the technologies kept."""
    common = ["EPCAM", "PTPRC"] + [f"S{i}" for i in range(248)]
    fams = {
        "spot": _family(["v1", "v2"], common + [f"V{i}" for i in range(50)], "Visium", counts=400.0),
        "imaging_cell": _members(
            [
                ("x1", "Xenium", common),
                ("x2", "Xenium", common),
                ("m1", "Merscope", common[:150]),
                ("m2", "Merscope", common[:150]),
            ],
            counts=400.0,
        ),
    }
    fams["imaging_cell"].X[fams["imaging_cell"].obs["technology"].to_numpy() == "Merscope"] = 0
    gates = joint_gates(fams, MARKERS, floor=50)
    assert "Merscope" in gates["gate3"]["excluded"]
    assert gates["gate1"]["n_shared_all"] == 150
    assert gates["gate1"]["n_shared"] == 250
    assert gates["gate1"]["passed"] is True


# mixing


def _points(labels, centres):
    """Observations around ``centres`` (one row per observation) with ``labels`` as their batch."""
    X = np.asarray(centres, dtype=float) + RNG.normal(0, 0.1, (len(labels), 2))
    return ad.AnnData(obs=pd.DataFrame({"b": labels}, index=[f"o{i}" for i in range(len(labels))]), obsm={"X_pca": X})


def test_mixing_separated_batches_score_zero():
    """Two far-apart batches: entropy and other-batch share are about 0."""
    labels = np.repeat(["a", "b"], 100)
    s = mixing_stats(_points(labels, np.repeat([[0, 0], [100, 100]], 100, axis=0)), "b", "X_pca")
    assert s["entropy"]["mean"] < 0.01
    assert s["other_share"]["mean"] < 0.01
    assert s["k"] == 30
    assert s["n_without_neighbours"] == 0


def test_mixing_random_labels_match_permutation_reference():
    """Random labels on the same points give values within 0.05 of the permutation reference."""
    labels = RNG.choice(["a", "b", "c"], 600, p=[0.5, 0.3, 0.2])
    s = mixing_stats(_points(labels, RNG.normal(0, 1, (600, 2))), "b", "X_pca")
    for name in ("entropy", "other_share"):
        assert abs(s[name]["mean"] - s[name]["random"]) < 0.05
        assert set(s[name]["random_per_batch"]) == {"a", "b", "c"}
    assert 0 < s["entropy"]["random_limit"] <= 1


def test_mixing_one_batch_is_undefined():
    """With one batch both statistics are undefined: None, with a note."""
    s = mixing_stats(_points(["a"] * 50, RNG.normal(0, 1, (50, 2))), "b", "X_pca")
    assert s["entropy"]["mean"] is None
    assert s["other_share"]["mean"] is None
    assert "one" in s["note"]


def test_host_reads_numpy_and_device_arrays():
    """_host returns numpy arrays as they are and copies device arrays (anything with .get()) to the host."""

    class Device:
        __cuda_array_interface__ = {}

        def get(self):
            return np.ones(3)

    np.testing.assert_array_equal(_host(np.zeros(2)), np.zeros(2))
    np.testing.assert_array_equal(_host(Device()), np.ones(3))
    frame = pd.DataFrame({"x": [1.0, 2.0]})  # has .get(key) but is not a device array
    np.testing.assert_array_equal(_host(frame), np.array([[1.0], [2.0]]))


# export


def test_embedding_rows_align_with_obs(tmp_path):
    """Each observation's embedding row in embedding.h5ad is the one its view computed."""
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    views = embed_views(fams, MARKERS, floor=10, use_gpu=False)
    e = ad.read_h5ad(_bundle(tmp_path, views=views) / "embedding.h5ad")
    for view, name in (("imaging_cell", "x2-7"), ("spot", "v1-3")):
        row = e.obs_names.get_loc(name)
        want = views[view].obsm["X_umap"][views[view].obs_names.get_loc(name)]
        np.testing.assert_allclose(e.obsm["X_umap"][row], want, rtol=1e-2, atol=1e-2)
        jrow = views["joint"].obs_names.get_loc(name)
        np.testing.assert_allclose(
            e.obsm["X_umap_joint"][row], views["joint"].obsm["X_umap"][jrow], rtol=1e-2, atol=1e-2
        )
    for v in ("spot", "imaging_cell", "joint"):
        assert e.obs[f"in_web_subsample_{v}"].dtype == bool
    assert e.obs["in_web_subsample_spot"].sum() == 120
    assert not e.obs.loc[e.obs["view"] == "imaging_cell", "in_web_subsample_spot"].any()


def test_manifest_mixing_genes_and_sizes(tmp_path):
    """Mixing by section and paper (and technology for the joint view), the gene rule, sizes and coverage."""
    m = json.loads((_bundle(tmp_path) / "manifest.json").read_text())
    assert set(m["mixing"]["spot"]) == {"section", "paper"}
    assert set(m["mixing"]["joint"]) == {"section", "paper", "technology"}
    assert m["views"]["spot"]["genes"]["measured"] == len(SPOT)
    assert m["views"]["spot"]["genes"]["size_factors"] == "all measured genes"
    assert m["embedding_h5ad_bytes"] == (tmp_path / "test-cohort" / "embedding.h5ad").stat().st_size
    assert m["member_genes"] == {
        "v1": {"n_genes": 350, "n_ensembl": 350},
        "v2": {"n_genes": 350, "n_ensembl": 350},
        "x1": {"n_genes": 270, "n_ensembl": 0},
        "x2": {"n_genes": 270, "n_ensembl": 0},
    }
    assert m["views_skipped"] == {}


def test_umap_table_keeps_nulls_and_integer_counts(tmp_path):
    """A missing paper stays null (not "nan"), and total_counts and n_genes are integers."""
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    fams["spot"].obs.loc[fams["spot"].obs["dataset_uid"] == "v1", "paper"] = None
    views = embed_views(fams, MARKERS, floor=10, use_gpu=False)
    umap = pd.read_parquet(_bundle(tmp_path, views=views) / "umap.parquet")
    spot = umap[umap["view"] == "spot"]
    assert spot.loc[spot["dataset_uid"] == "v1", "paper"].isna().all()
    assert "nan" not in set(umap["paper"].dropna())
    assert umap["total_counts"].dtype.kind == "i"
    assert umap["n_genes"].dtype.kind == "i"


def _qc_dropping(uids):
    return pd.DataFrame(
        {
            "dataset_uid": ["v1", "v2", "x1", "x2"],
            "n_before": [60] * 4,
            "n_after": [0 if u in uids else 60 for u in ("v1", "v2", "x1", "x2")],
            "flagged": [False] * 4,
            "status": ["dropped: no cells after QC" if u in uids else "kept" for u in ("v1", "v2", "x1", "x2")],
        }
    )


def test_qc_dropped_member_leaves_members_kept(tmp_path):
    """A member with no cells after QC is listed as dropped in the manifest and the cohort table."""
    out = _bundle(tmp_path, qc=_qc_dropping({"x2"}))
    m = json.loads((out / "manifest.json").read_text())
    assert "x2" not in m["members_kept"]
    assert m["members_dropped"]["x2"] == "dropped: no cells after QC"
    assert m["n_datasets"] == 3
    assert m["n_papers"] == 3
    table = pd.read_parquet(out / "cohort.parquet").set_index("uid")
    assert table.loc["x2", "status"] == "dropped: no cells after QC"


def test_refuses_fewer_than_3_papers_after_qc(tmp_path):
    """Fewer than 3 papers left after QC is refused, naming the count."""
    with pytest.raises(ValueError, match="2 papers"):
        _bundle(tmp_path, qc=_qc_dropping({"x1", "x2"}))


def test_split_views_in_manifest(tmp_path):
    """Per-technology views carry their family and note, and a skipped technology is listed with its reason."""
    common = [f"S{i}" for i in range(150)]
    fam = _members(
        [
            ("x1", "Xenium", common + [f"X{i}" for i in range(100)]),
            ("x2", "Xenium", common + [f"X{i}" for i in range(100)]),
            ("v1", "Merscope", common + [f"M{i}" for i in range(150)]),
            ("v2", "MERFISH", common[:40]),
        ]
    )
    views = embed_views({"imaging_cell": fam}, MARKERS, floor=0, use_gpu=False)
    out = _bundle(tmp_path, views=views)
    m = json.loads((out / "manifest.json").read_text())
    assert m["views"]["imaging_cell:Xenium"]["family"] == "imaging_cell"
    assert "= 40 shared genes (under 200)" in m["views"]["imaging_cell:Xenium"]["note"]
    assert "under 50" in m["views_skipped"]["imaging_cell:MERFISH"]
    e = ad.read_h5ad(out / "embedding.h5ad")
    assert "in_web_subsample_imaging_cell_Xenium" in e.obs


def test_embedding_single_part_listed(tmp_path):
    """Under the target, the bundle has one embedding.h5ad, and the manifest lists it as the only part."""
    out = _bundle(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    (part,) = m["embedding"]
    assert part["file"] == "embedding.h5ad"
    assert part["bytes"] == (out / "embedding.h5ad").stat().st_size == m["embedding_h5ad_bytes"]
    assert part["n_obs"] == ad.read_h5ad(out / "embedding.h5ad").n_obs
    assert sorted(part["dataset_uids"]) == ["v1", "v2", "x1", "x2"]


def test_embedding_split_by_whole_datasets_and_concat_rebuilds_it(tmp_path):
    """Over the target, the embedding is split into parts of whole datasets; concat over them rebuilds it."""
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    views = embed_views(fams, MARKERS, floor=10, use_gpu=False)
    whole = ad.read_h5ad(_bundle(tmp_path / "whole", views=views) / "embedding.h5ad")
    size = (tmp_path / "whole" / "test-cohort" / "embedding.h5ad").stat().st_size
    out = _bundle(
        tmp_path / "split", views=views, embedding_target=int(size * 0.85)
    )  # a one-dataset part is ~70% overhead here
    m = json.loads((out / "manifest.json").read_text())
    files = [p["file"] for p in m["embedding"]]
    assert len(files) >= 2
    assert files == [f"embedding-{i}.h5ad" for i in range(1, len(files) + 1)]
    assert not (out / "embedding.h5ad").exists()
    parts = [ad.read_h5ad(out / f) for f in files]
    for meta, part in zip(m["embedding"], parts, strict=True):
        assert meta["n_obs"] == part.n_obs
        assert meta["bytes"] == (out / meta["file"]).stat().st_size
        assert sorted(meta["dataset_uids"]) == sorted(part.obs["dataset_uid"].astype(str).unique())
    uids = [set(p["dataset_uids"]) for p in m["embedding"]]
    assert all(not (a & b) for i, a in enumerate(uids) for b in uids[i + 1 :])  # a dataset is never split
    rebuilt = ad.concat(parts, merge="same")
    assert list(rebuilt.obs_names) == list(whole.obs_names)
    pd.testing.assert_frame_equal(rebuilt.obs, whole.obs, check_categorical=False)
    assert set(rebuilt.obsm) == set(whole.obsm)
    for key in whole.obsm:
        np.testing.assert_array_equal(np.asarray(rebuilt.obsm[key]), np.asarray(whole.obsm[key]))
    assert m["embedding_h5ad_bytes"] == sum(p["bytes"] for p in m["embedding"])


def test_dataset_alone_over_embedding_target_refused(tmp_path):
    """A dataset that alone is over the target can't be split, so the export stops and leaves no h5ad."""
    with pytest.raises(ValueError, match="alone"):
        _bundle(tmp_path, embedding_target=10)
    assert not list(tmp_path.rglob("*.h5ad"))


def test_normalize_total_sees_all_measured_genes(monkeypatch):
    """normalize_total runs on all 2,500 measured genes, before the HVG subset."""
    import scanpy as sc

    seen = []
    real = sc.pp.normalize_total

    def spy(a, *args, **kwargs):
        seen.append(a.n_vars)
        return real(a, *args, **kwargs)

    monkeypatch.setattr(sc.pp, "normalize_total", spy)
    genes = ["EPCAM", "PTPRC"] + [f"G{i}" for i in range(2498)]
    v = embed_views({"spot": _family(["v1", "v2"], genes, "Visium", counts=3000.0)}, MARKERS, floor=0, use_gpu=False)[
        "spot"
    ]
    assert seen == [2500]
    assert v.n_vars == 2000


def test_neighbour_stats_match_brute_force():
    """The CSR-based entropy and other-share equal a dense per-row count."""
    from primers.embed import _neighbour_stats

    rng = np.random.default_rng(1)
    n, k, n_b = 200, 7, 4
    rows = np.repeat(np.arange(n), k)
    cols = np.concatenate([rng.choice(np.delete(np.arange(n), i), k, replace=False) for i in range(n)])
    import scipy.sparse as sp

    graph = sp.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
    codes = rng.integers(0, n_b, n)
    entropy, other = _neighbour_stats(graph, codes, n_b)
    for i in range(n):
        counts = np.bincount(codes[cols[i * k : (i + 1) * k]], minlength=n_b)
        p = counts[counts > 0] / k
        assert entropy[i] == pytest.approx(-(p * np.log(p)).sum() / np.log(min(k, n_b)))
        assert other[i] == pytest.approx(1 - counts[codes[i]] / k)


def test_single_technology_family_keeps_its_name():
    """A one-technology family under 200 shared genes stays one view under its family name."""
    small = ["EPCAM", "PTPRC"] + [f"S{i}" for i in range(98)]
    views = embed_views({"imaging_cell": _family(["x1", "x2"], small, "Xenium")}, MARKERS, floor=0, use_gpu=False)
    assert set(views) == {"imaging_cell"}
    assert views["imaging_cell"].uns["primer"]["note"] == "only 100 shared genes (under 200)"


def test_no_view_left_raises():
    """When every technology has fewer than 50 shared genes, embedding raises instead of returning nothing."""
    tiny = ["EPCAM", "PTPRC"] + [f"S{i}" for i in range(20)]
    with pytest.raises(ValueError, match="no view left"):
        embed_views({"imaging_cell": _family(["x1", "x2"], tiny, "Xenium")}, MARKERS, floor=0, use_gpu=False)


def _member_genes(adata):
    adata.uns["member_genes"] = {u: list(adata.var_names) for u in adata.obs["dataset_uid"].astype(str).unique()}
    return adata


def test_tiny_panel_dropped_before_it_knocks_out_xenium():
    """A 40-gene MERFISH panel is dropped first; without that, the 40 shared genes would push Xenium under its floor."""
    merfish_genes = SHARED[:40]
    imaging = ad.concat(
        [
            _family(["x1", "x2"], IMAGING, "Xenium", counts=50.0),
            _family(["f1"], merfish_genes, "MERFISH", counts=200.0),
        ],
        join="outer",
    )
    imaging.uns["member_genes"] = {"x1": IMAGING, "x2": IMAGING, "f1": merfish_genes}
    fams = {"spot": _member_genes(_family(["v1", "v2"], SPOT, "Visium", counts=3000.0)), "imaging_cell": imaging}
    floors = {"Visium": 100, "Xenium": 10, "MERFISH": 10}
    gates = joint_gates(fams, MARKERS, floor=floors)
    assert [d["technology"] for d in gates["drops"]] == ["MERFISH"]
    assert gates["drops"][0]["step"] == "panel"
    assert "Xenium" not in gates["gate3"]["excluded"]
    assert gates["joint"] is True
    assert gates["gate1"]["n_shared"] == len(SHARED)
    # on the 40 genes alone, Xenium (50 counts over 270 genes) would fall below its floor of 10
    from primers.qc import floor_loss

    assert (
        floor_loss(fams["imaging_cell"][fams["imaging_cell"].obs["technology"] == "Xenium"], 10, genes=merfish_genes)[
            "Xenium"
        ]
        > 0.5
    )


def test_smallest_panel_dropped_until_200_shared():
    """With fewer than 200 shared genes, the smallest panel is dropped and the gates are tried again."""
    small = SHARED[:120] + [f"Q{i}" for i in range(10)]
    imaging = ad.concat([_family(["x1", "x2"], IMAGING, "Xenium"), _family(["m1"], small, "Merscope")], join="outer")
    imaging.uns["member_genes"] = {"x1": IMAGING, "x2": IMAGING, "m1": small}
    fams = {"spot": _member_genes(_family(["v1", "v2"], SPOT, "Visium")), "imaging_cell": imaging}
    gates = joint_gates(fams, MARKERS, floor=0)
    assert [(d["technology"], d["step"]) for d in gates["drops"]] == [("Merscope", "gate1")]
    assert "smallest panel (130 genes)" in gates["drops"][0]["reason"]
    assert gates["joint"] is True


def test_floors_in_manifest(tmp_path):
    """The per-technology floors used by gate 3 and the joint view are stated in the manifest."""
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    views = embed_views(fams, MARKERS, floor={"Visium": 10, "Xenium": 5}, use_gpu=False)
    out = _bundle(tmp_path, views=views)
    m = json.loads((out / "manifest.json").read_text())
    assert m["floors"] == {"Visium": 10, "Xenium": 5}
    assert m["gates"]["drops"] == []


def test_export_uses_mixing_computed_by_the_caller(tmp_path):
    """A ``mixing`` dict passed to export_bundle goes into the manifest as given, so it is computed once."""
    fams = {"spot": _family(["v1", "v2"], SPOT, "Visium"), "imaging_cell": _family(["x1", "x2"], IMAGING, "Xenium")}
    views = embed_views(fams, MARKERS, floor=10, use_gpu=False)
    mixing = mixing_by_view(views)
    assert set(mixing) == set(views)
    assert set(mixing["joint"]) == {"section", "paper", "technology"}
    sentinel = {"spot": {"section": {"X_pca": {"note": "from the caller"}}}}
    m = json.loads((_bundle(tmp_path, views=views, mixing=sentinel) / "manifest.json").read_text())
    assert m["mixing"] == sentinel


def test_manifest_records_primers_ref(tmp_path):
    """The manifest names the code that made the bundle."""
    m = json.loads((_bundle(tmp_path) / "manifest.json").read_text())
    assert m["primers_ref"] == "abc123"


def test_manifest_records_the_full_spec(tmp_path):
    """The manifest keeps include_uids, exclude_uids and families next to the query (for the site's staleness)."""
    out = _bundle(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    assert set(m["spec"]) == {"query", "include_uids", "exclude_uids", "families"}
    assert m["spec"]["query"] == m["query"]
