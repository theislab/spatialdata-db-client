from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from sddb.adapter import build, check
from sddb.objectmeta import ObjectMeta


class FakeAdapter:
    name = "fake"
    version = "0.1"
    config = {"k": 1}

    def requirements(self):
        return {"tables": ["table"]}

    def validate_meta(self, meta):
        return [] if "tables/table" in meta.elements else ["missing table"]

    def validate(self, sdata):
        return []

    def build(self, sdata):
        return {"n": 1}


def _meta(uid, has_table):
    el = {"tables/table": {}} if has_table else {"images/img": {}}
    return ObjectMeta(uid, f"z/{uid}", None, el, ("table",) if has_table else (), "sha256:x")


def test_check_advisory_tier_counts_and_reasons(monkeypatch):
    members = pd.DataFrame({"uid": ["a", "b"], "zarr_url": ["za", "zb"], "fingerprint": ["x", "y"]})
    monkeypatch.setattr("sddb.adapter._metas", lambda m, **k: [_meta("a", True), _meta("b", False)])
    rep = check(members, FakeAdapter(), deep=False)
    assert rep.total == 2
    assert rep.compatible == 1
    assert rep.tier == "advisory"
    assert "b" in rep.excluded["missing table"]


def test_check_deep_uses_authoritative_validate(monkeypatch):
    members = pd.DataFrame({"uid": ["a"], "zarr_url": ["za"], "fingerprint": ["x"]})
    monkeypatch.setattr("sddb.adapter._open", lambda url, a: object())
    rep = check(members, FakeAdapter(), deep=True)
    assert rep.tier == "authoritative"
    assert rep.compatible == 1


def _members():
    return pd.DataFrame({"uid": ["a"], "zarr_url": ["za"], "fingerprint": ["x"], "default_table": ["table"]})


def test_build_materialize_writes_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("sddb.adapter._open", lambda url, a: object())  # skip real sdata open
    build(_members(), FakeAdapter(), materialize=True, out=tmp_path, verify=False, cohort_hash="h")
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["cohort_hash"] == "h"
    assert prov["split"] is None
    assert prov["created_at"]
    assert prov["adapter"] == {"name": "fake", "version": "0.1", "config": {"k": 1}}
    assert "sddb_version" in prov
    assert prov["members"][0] == {"uid": "a", "table": "table", "output": "dict", "elements": None}


def test_build_lazy_opens_nothing(monkeypatch):
    def boom(url, a):
        raise AssertionError("opened")

    monkeypatch.setattr("sddb.adapter._open", boom)
    view = build(_members(), FakeAdapter(), verify=False, cohort_hash="h")
    assert len(view) == 1


def test_build_split_parent_hash_mismatch_raises():
    split = SimpleNamespace(parent_hash="other")
    with pytest.raises(ValueError, match="parent_hash"):
        build(_members(), FakeAdapter(), split=split, verify=False, cohort_hash="h")


def test_build_split_match_records_split_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("sddb.adapter._open", lambda url, a: object())
    split = SimpleNamespace(parent_hash="h", by="study_id", seed=7, ratios={"train": 0.8}, mode="group")
    build(_members(), FakeAdapter(), split=split, materialize=True, out=tmp_path, verify=False, cohort_hash="h")
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["cohort_hash"] == "h"
    assert prov["split"]["seed"] == 7
    assert prov["split"] == {"by": "study_id", "seed": 7, "ratios": {"train": 0.8}, "mode": "group"}
    assert "split_hash" not in prov


def test_build_verify_runs_before_open(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("sddb.freeze.verify_members", lambda m, **k: calls.append("verify"))
    monkeypatch.setattr("sddb.adapter._open", lambda url, a: calls.append("open") or object())
    build(_members(), FakeAdapter(), materialize=True, out=tmp_path, verify=True, cohort_hash="h")
    assert calls == ["verify", "open"]


def test_build_verify_false_skips_verify(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("sddb.freeze.verify_members", lambda m, **k: calls.append("verify"))
    monkeypatch.setattr("sddb.adapter._open", lambda url, a: calls.append("open") or object())
    build(_members(), FakeAdapter(), materialize=True, out=tmp_path, verify=False, cohort_hash="h")
    assert "verify" not in calls
    assert calls == ["open"]
