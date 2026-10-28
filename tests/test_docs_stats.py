import importlib.util
import logging
from pathlib import Path

import pandas as pd

import sddb

_PATH = Path(__file__).parent.parent / "docs" / "_ext" / "catalog_stats.py"
_spec = importlib.util.spec_from_file_location("catalog_stats", _PATH)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _df(extra=()):
    rows = [
        ("A big", "big", 900, "pass", "Xenium", "s1", "human"),
        ("A small", "small", 100, "pass", "Xenium", "s1", "human"),
        ("Vis", "vis", 50, "pass", "Visium", "s2", "mouse"),
        ("ST", "st", 70, "pass", "SpatialTranscriptomics", "s3", "human"),
        *extra,
    ]
    cols = ["title", "slug", "size_bytes", "validation_status", "technology", "study_id", "organism"]
    return pd.DataFrame(rows, columns=cols)


def test_stats_from_frame():
    s = cs.catalog_stats(_df())
    assert s["n_datasets"] == "4"
    assert s["n_studies"] == "3"
    assert s["n_organisms"] == "2"
    assert s["n_Xenium"] == "2"
    assert s["example_Xenium"] == "[A small](https://spatialdata-db-web.pages.dev/datasets/small/)"


def test_unknown_technology_listed():
    df = _df([("New", "new", 1, "pass", "NewTech", "s4", "human")])
    assert ("NewTech", 1) in cs.assay_rows(df)
    assert cs.assay_rows(df)[0] == ("Xenium", 2)


def test_load_stats_offline_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(sddb, "Catalog", boom)
    assert cs.load_stats() == {}


def test_technology_token_has_no_spaces():
    df = _df([("X", "x", 5, "pass", "spatial transcriptomics", "s5", "human")])
    s = cs.catalog_stats(df)
    assert s["n_SpatialTranscriptomics"] == "2"
    assert s["example_SpatialTranscriptomics"].endswith("/datasets/x/)")


def test_nan_size_bytes_does_not_crash():
    df = _df([("Nan", "nan", float("nan"), "pass", "Atera", "s6", "human")])
    s = cs.catalog_stats(df)
    assert s["n_Atera"] == "1"
    assert "example_Atera" not in s
    assert s["example_Xenium"].endswith("/datasets/small/)")


def test_other_assays_lists_only_undescribed():
    df = _df([("New", "new", 1, "pass", "NewTech", "s4", "human")])
    s = cs.catalog_stats(df)
    assert s["other_assays"] == "**Other technologies**\n\n- **NewTech**: 1 dataset. No description yet."
    assert cs.catalog_stats(_df())["other_assays"] == ""


def test_load_stats_warns_on_failure(monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(sddb, "Catalog", boom)
    with caplog.at_level(logging.WARNING):
        assert cs.load_stats() == {}
        cs.warn_if_unavailable()
    assert "offline" in caplog.text


def test_with_defaults_blanks_missing():
    d = cs.with_defaults({})
    assert d["intro_count"] == ""
    assert d["line_Visium"] == ""
    assert d["other_assays"] == ""


def test_line_and_tier_note():
    s = cs.catalog_stats(_df().assign(tier=["silver", "silver", None, "bronze"]))
    assert s["line_Visium"].startswith("**1 dataset**")
    assert "2 datasets (silver)" in s["tier_note"]
    assert "1 dataset (unassigned)" in s["tier_note"]


def test_other_assays_aggregates_by_token_and_categorical_tier():
    df = _df(
        [
            ("N1", "n1", 1, "pass", "new tech", "s4", "human"),
            ("N2", "n2", 2, "pass", "newTech", "s4", "human"),
        ]
    ).assign(tier=pd.Categorical(["silver"] * 6))
    s = cs.catalog_stats(df)
    assert s["other_assays"].count("No description yet.") == 1
    assert "2 datasets" in s["other_assays"]
    assert "6 datasets (silver)" in s["tier_note"]


def test_catalog_line_with_and_without_date():
    assert cs.catalog_stats(_df())["catalog_line"] == "The catalog holds 4 datasets from 3 technologies."
    dated = cs.catalog_stats(_df(), "2026-10-28T09:00:00")["catalog_line"]
    assert dated == "The catalog of 2026-10-28 holds 4 datasets from 3 technologies."
    one = cs.catalog_stats(_df().assign(assay="Visium Spatial Gene Expression"))["catalog_line"]
    assert one == "The catalog holds 4 datasets from 1 technology."  # counts `assay`, like the website
    assert cs.with_defaults({})["catalog_line"] == ""


def test_site_url_single_source():
    assert cs.SITE_URL.startswith("https://")
    root = Path(__file__).parent.parent
    skip = {"design", "superpowers", "_build"}
    hits = []
    for base in (root / "docs", root / "tests"):
        for p in base.rglob("*"):
            if not p.is_file() or skip & set(p.relative_to(root).parts):
                continue
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                continue
            if "spatialdata-db" + ".com" in text:
                hits.append(str(p))
    assert not hits, hits


def test_example_label_uses_organism_and_tissue():
    df = _df().assign(tissue="lung")
    assert cs.catalog_stats(df)["example_Xenium"].startswith("[human lung](")


def test_small_examples_match_website_rule():
    df = _df().assign(uid=["u" * 20, "s" * 20, "v" * 20, "t" * 20])
    df["collections"] = [["c1"], ["c1", "c2"], ["c2"], None]
    s = cs.catalog_stats(df)
    assert s["small_uid"] == "v" * 20  # size 50, the smallest
    assert s["small_stem"] == "v" * 16
    assert s["small_collection"] == "c2"  # 150 bytes total, against 1000 for c1


def test_small_examples_have_offline_fallback():
    d = cs.with_defaults({})
    assert d["small_uid"] == "<uid>"
    assert d["small_collection"] == "<slug>"
    assert "small_uid" not in cs.catalog_stats(_df())  # no uid column: nothing to compute


def test_fill_code_examples_replaces_placeholders():
    class App:
        class config:
            myst_substitutions = cs.with_defaults({"small_uid": "u1", "small_collection": "c1"})

    src = ['Catalog().dataset("{{ small_uid }}") {{ small_collection }} {{ small_stem }} [x]({{ site_url }})']
    cs.fill_code_examples(App, "start", src)
    assert src[0] == f'Catalog().dataset("u1") c1 <16-character stem> [x]({cs.SITE_URL})'


def test_start_page_has_no_unfilled_placeholder():
    class App:
        class config:
            myst_substitutions = cs.with_defaults({})

    raw = (Path(__file__).parent.parent / "docs" / "start.md").read_text()
    src = [raw]
    cs.fill_code_examples(App, "start", src)
    assert "{{" not in src[0]


def test_small_collection_skips_collections_with_unsized_members():
    df = _df().assign(uid=["u" * 20, "s" * 20, "v" * 20, "t" * 20])
    df["collections"] = [["c1"], ["c1"], ["c2"], ["c2"]]
    df.loc[3, "size_bytes"] = float("nan")
    assert cs.catalog_stats(df)["small_collection"] == "c1"
