from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from sddb.freeze import FrozenCohort
from sddb.split import SplitManifest, _largest_remainder, make_split


def _members(studies):
    return pd.DataFrame({"uid": [f"u{i}" for i in range(len(studies))], "study_id": studies})


def test_deterministic_and_group_safe():
    m = _members(["A", "A", "B", "C", "D", "E", "F", "G", "H", "I"])
    s1 = make_split(m, "h", train=0.8, val=0.1, test=0.1, seed=42)
    s2 = make_split(m, "h", train=0.8, val=0.1, test=0.1, seed=42)
    assert s1.assignments == s2.assignments
    by_study: dict[str, set[str]] = {}
    for u, st in zip(m["uid"], m["study_id"], strict=True):
        by_study.setdefault(st, set()).add(s1.assignments[u])
    assert all(len(f) == 1 for f in by_study.values())
    assert set(s1.assignments.values()) == {"train", "val", "test"}


def test_row_order_does_not_change_split():
    m = _members(["A", "A", "B", "C", "D", "E", "F", "G", "H", "I"])
    s1 = make_split(m, "h", train=0.6, val=0.2, test=0.2, seed=7)
    s2 = make_split(m.iloc[::-1].reset_index(drop=True), "h", train=0.6, val=0.2, test=0.2, seed=7)
    assert s1.assignments == s2.assignments


def test_missing_study_is_singleton():
    m = pd.DataFrame({"uid": ["u0", "u1"], "study_id": [pd.NA, "A"]})
    s = make_split(m, "h", train=0.5, val=0.0, test=0.5, seed=1)
    assert set(s.assignments) == {"u0", "u1"}
    assert s.assignments["u0"] != s.assignments["u1"]
    a = make_split(m, "h", train=0.5, val=0.0, test=0.5, seed=1, mode="annotate")
    assert a.assignments["u0"] == "object:u0"


def test_too_few_groups_raises():
    m = _members(["A", "A"])
    with pytest.raises(ValueError, match="1 group"):
        make_split(m, "h", train=0.34, val=0.33, test=0.33, seed=1)


def test_annotate_mode_labels_only():
    m = _members(["A", "B"])
    s = make_split(m, "h", train=0.5, val=0.0, test=0.5, seed=1, mode="annotate")
    assert s.assignments == {"u0": "A", "u1": "B"}


@pytest.mark.parametrize("n", range(3, 40))
def test_largest_remainder_sums_and_nonempty(n):
    c = _largest_remainder(n, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert sum(c.values()) == n
    assert all(v >= 1 for v in c.values())


def test_write_load_roundtrip_and_cohort_split(tmp_path):
    m = _members(["A", "B", "C", "D"])
    fc = SimpleNamespace(members=m, hash="sha256:abc")
    s = FrozenCohort.split(fc, train=0.5, val=0.25, test=0.25, seed=3)  # type: ignore[arg-type]
    assert s.parent_hash == "sha256:abc"
    assert SplitManifest.load(s.write(tmp_path / "split.json")) == s
