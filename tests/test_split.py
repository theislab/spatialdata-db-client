from __future__ import annotations

import pandas as pd
import pytest

from sddb.split import _largest_remainder, make_split


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


def test_missing_by_column_raises():
    with pytest.raises(KeyError, match="does_not_exist"):
        make_split(_members(["A", "B"]), "h", by="does_not_exist", train=0.5, val=0.0, test=0.5, seed=1)


def test_non_identifier_by_column_works():
    m = pd.DataFrame({"uid": ["u0", "u1", "u2"], "study-id": ["A", "A", "B"]})
    s = make_split(m, "h", by="study-id", train=0.5, val=0.0, test=0.5, seed=1, mode="annotate")
    assert s.assignments == {"u0": "A", "u1": "A", "u2": "B"}


@pytest.mark.parametrize("r", [(0.5, 0.1, 0.1), (1.5, -0.5, 0.0), (0.0, 0.0, 0.0)])
@pytest.mark.parametrize("mode", ["group", "annotate"])
def test_bad_ratios_raise(r, mode):
    with pytest.raises(ValueError, match="ratios"):
        make_split(_members(["A", "B", "C"]), "h", train=r[0], val=r[1], test=r[2], seed=1, mode=mode)


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        make_split(_members(["A", "B"]), "h", train=0.5, val=0.0, test=0.5, seed=1, mode="nope")


def _series(row):
    return row["series"]


def test_by_callable_groups_by_its_key():
    """A GEO study_id is one sample (GSM), so the default splits a series; a callable groups by series."""
    m = pd.DataFrame(
        {
            "uid": [f"u{i}" for i in range(6)],
            "study_id": [f"geo:GSM{i}" for i in range(6)],
            "series": ["GSE1", "GSE1", "GSE1", "GSE2", "GSE2", None],
        }
    )
    s = make_split(m, "h", by=_series, train=0.5, val=0.0, test=0.5, seed=3)
    assert s.by == "_series"
    assert len({s.assignments[u] for u in ("u0", "u1", "u2")}) == 1  # one series, one fold
    assert s.assignments["u3"] == s.assignments["u4"]
    a = make_split(m, "h", by=_series, train=0.5, val=0.0, test=0.5, seed=3, mode="annotate")
    assert a.assignments == {"u0": "GSE1", "u1": "GSE1", "u2": "GSE1", "u3": "GSE2", "u4": "GSE2", "u5": "object:u5"}
    assert make_split(m, "h", by=lambda r: r["series"], train=0.5, val=0.0, test=0.5, seed=3).by == "callable"


def test_by_column_matches_the_same_callable():
    m = _members(["A", "A", "B", "C", "D", "E"])
    col = make_split(m, "h", by="study_id", train=0.5, val=0.25, test=0.25, seed=9)
    fn = make_split(m, "h", by=lambda r: r["study_id"], train=0.5, val=0.25, test=0.25, seed=9)
    assert col.by == "study_id"
    assert col.assignments == fn.assignments


def test_by_rejects_other_types():
    with pytest.raises(TypeError, match="column name or a callable"):
        make_split(_members(["A", "B"]), "h", by=3, train=0.5, val=0.0, test=0.5, seed=1)
