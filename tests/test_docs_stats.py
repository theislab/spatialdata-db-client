import importlib.util
from pathlib import Path

import pandas as pd
import pytest

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
    assert s["example_Xenium"] == "[A small](https://spatialdata-db.com/datasets/small/)"


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
    assert s["n_SpatialTranscriptomics"] == "1"
    assert s["example_SpatialTranscriptomics"].endswith("/datasets/x/)")
