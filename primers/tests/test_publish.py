import importlib.util
import json
from pathlib import Path

import pytest
from conftest import member

ROOT = Path(__file__).resolve().parents[1]


def _pub():
    """Import scripts/publish_primers.py, which is not part of the package."""
    spec = importlib.util.spec_from_file_location("publish_primers", ROOT / "scripts" / "publish_primers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundle(root, slug, *, release="2026-10-28", path="gpu", dev_note="", files=None):
    b = root / slug
    b.mkdir(parents=True)
    (b / "manifest.json").write_text(json.dumps({"slug": slug, "release": release, "path": path}))
    for f in files or ("umap.parquet", "qc.parquet", "cohort.parquet", "embedding.h5ad", "primer.html"):
        (b / f).write_bytes(b"x" * 10)
    nb = {"metadata": {"papermill": {"parameters": {"slug": slug, "dev_note": dev_note}}}, "cells": []}
    (b / "primer.ipynb").write_text(json.dumps(nb))
    return b


def test_ready_bundle_staged_in_the_site_shape(tmp_path):
    """Assets are flat <slug>__<file>; index.json lists each slug's files with sizes, as the site reads it."""
    pub = _pub()
    b = _bundle(tmp_path / "bundles", "human-colon")
    assert pub.check_bundle(b, "2026-10-28") == []
    out = tmp_path / "stage"
    index = pub.stage([b], "2026-10-28", out)
    assert index["tag"] == "primers-2026-10-28"
    assert index["primers"][0]["slug"] == "human-colon"
    files = index["primers"][0]["files"]
    assert set(files) == {
        "manifest.json",
        "umap.parquet",
        "qc.parquet",
        "cohort.parquet",
        "embedding.h5ad",
        "primer.ipynb",
        "primer.html",
    }
    assert files["umap.parquet"] == 10
    assert (out / "human-colon__umap.parquet").read_bytes() == b"x" * 10
    assert json.loads((out / "index.json").read_text()) == index
    cmds = pub.commands(index, out)
    assert cmds[0].startswith("gh release create primers-2026-10-28 --repo theislab/spatialdata-db-client")
    assert "--generate-notes" in cmds[0]
    assert all(c.startswith("gh release upload primers-2026-10-28") for c in cmds[1:])
    assert len(cmds) == 1 + 1 + len(files)


def test_embedding_parts_staged_from_the_manifest(tmp_path):
    """A split embedding is staged part by part, as the manifest lists it; a missing part blocks publishing."""
    pub = _pub()
    b = _bundle(tmp_path / "bundles", "mouse-brain", files=("umap.parquet", "qc.parquet", "cohort.parquet"))
    parts = [{"file": f"embedding-{i}.h5ad", "bytes": 10, "n_obs": 1, "dataset_uids": [f"u{i}"]} for i in (1, 2)]
    m = json.loads((b / "manifest.json").read_text())
    (b / "manifest.json").write_text(json.dumps({**m, "embedding": parts}))
    (b / "embedding-1.h5ad").write_bytes(b"x" * 10)
    assert pub.check_bundle(b, "2026-10-28") == ["misses embedding-2.h5ad"]
    (b / "embedding-2.h5ad").write_bytes(b"x" * 10)
    assert pub.check_bundle(b, "2026-10-28") == []
    files = pub.stage([b], "2026-10-28", tmp_path / "stage")["primers"][0]["files"]
    assert {"embedding-1.h5ad", "embedding-2.h5ad"} <= set(files)
    assert "embedding.h5ad" not in files


@pytest.mark.parametrize(
    ("kwargs", "slug", "reason"),
    [
        ({}, "human-colon-smoke", "smoke run"),
        ({"path": "cpu"}, "a", "GPU run"),
        ({"release": "2026-10-16"}, "a", "not '2026-10-28'"),
        ({"dev_note": "Development run: ..."}, "a", "development run"),
        ({"files": ("umap.parquet", "qc.parquet")}, "a", "misses cohort.parquet"),
    ],
)
def test_bundle_refused_with_reason(tmp_path, kwargs, slug, reason):
    """Smoke, CPU, wrong-release, development and incomplete bundles are refused, naming why."""
    pub = _pub()
    problems = pub.check_bundle(_bundle(tmp_path, slug, **kwargs), "2026-10-28")
    assert any(reason in p for p in problems), problems


def test_oversized_asset_refused(tmp_path, monkeypatch):
    """A file at or over GitHub's 2 GiB asset limit is refused."""
    pub = _pub()
    monkeypatch.setattr(pub, "MAX_ASSET_BYTES", 10)
    assert any("asset limit" in p for p in pub.check_bundle(_bundle(tmp_path, "a"), "2026-10-28"))


def test_main_stages_nothing_when_a_bundle_fails(tmp_path, monkeypatch, capsys):
    """One failing bundle stops the run before anything is staged."""
    pub = _pub()
    _bundle(tmp_path / "bundles", "good")
    _bundle(tmp_path / "bundles", "bad", path="cpu")
    monkeypatch.setattr(
        "sys.argv", ["x", str(tmp_path / "bundles"), "--release", "2026-10-28", "--out", str(tmp_path / "stage")]
    )
    assert pub.main() == 1
    assert not (tmp_path / "stage").exists()
    assert "bad:" in capsys.readouterr().err


def test_staleness_counts_added_and_removed(make_catalog):
    """Members the spec selects now but not then are added; members gone from the catalog are removed."""
    from primers.cohort import staleness

    cat = make_catalog([member(u) for u in ("a", "b", "c", "d")])
    manifest = {
        "spec": {"query": {}, "include_uids": None, "exclude_uids": ["d"], "families": None},
        "members_kept": ["a", "b"],
        "members_dropped": {"z": "dropped: licence not redistributable"},
    }
    s = staleness(manifest, cat)
    assert s == {"stale": True, "added": ["c"], "removed": ["z"]}
    fresh = {**manifest, "members_kept": ["a", "b", "c"], "members_dropped": {}}
    assert staleness(fresh, cat) == {"stale": False, "added": [], "removed": []}
