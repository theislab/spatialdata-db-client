from __future__ import annotations

import pandas as pd
from tests._fixtures import make_tiny_sdata_zarr

from sddb.cohort import SpatialDataCohort
from sddb.dataset import Source
from sddb.freeze import FrozenCohort, cohort_hash


def _members():
    return pd.DataFrame(
        {
            "uid": ["u2", "u1"],
            "lamin_version": [None, None],
            "fingerprint": ["sha256:b", "sha256:a"],
            "study_id": ["S2", "S1"],
            "zarr_url": ["z2", "z1"],
        }
    )


def test_hash_is_order_independent_and_excludes_timestamp():
    m = _members()
    h1 = cohort_hash({"organism": "human"}, m)
    h2 = cohort_hash({"organism": "human"}, m.iloc[::-1].reset_index(drop=True))
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_changes_with_membership():
    m = _members()
    assert cohort_hash({}, m) != cohort_hash({}, m.iloc[:1])


def test_write_load_roundtrip(tmp_path):
    fc = FrozenCohort(
        "1",
        "2026-10-07T00:00:00Z",
        Source(url="cat", version="v"),
        {"organism": "human"},
        cohort_hash({"organism": "human"}, _members()),
        _members(),
    )
    p = fc.write(tmp_path / "cohort.json")
    back = FrozenCohort.load(p)
    assert back.hash == fc.hash
    assert list(back.members["uid"]) == list(fc.members["uid"])
    assert back.filter == {"organism": "human"}


def test_write_load_sidecar_roundtrip(tmp_path):
    n = 250
    m = pd.DataFrame({"uid": [f"u{i}" for i in range(n)], "lamin_version": [None] * n, "fingerprint": ["sha256:x"] * n})
    fc = FrozenCohort("1", "t", Source(url="c", version="v"), {}, cohort_hash({}, m), m)
    p = fc.write(tmp_path / "c.json")
    assert (tmp_path / "c.members.parquet").exists()
    back = FrozenCohort.load(p)
    assert len(back.members) == n
    assert back.hash == fc.hash


def test_query_records_deterministic_filter(tmp_path):
    import json

    import pytest
    from tests._fixtures import write_fixture_catalog

    from sddb.catalog import Catalog

    p = write_fixture_catalog(tmp_path / "cat.parquet")
    cat = Catalog(p.as_uri(), cache_dir=tmp_path / "cache")
    a = cat.query(organism={"mouse", "human"})._filter
    b = cat.query(organism=("human", "mouse"))._filter
    assert a == b == {"validation": "pass", "organism": ["human", "mouse"]}
    json.dumps(a)
    assert cat.query(validation=None)._filter == {}
    with pytest.warns(UserWarning, match="nothing matched"):
        assert cat.search("zzzzqqq")._filter == {"search": "zzzzqqq"}


def test_freeze_local_store(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    df = pd.DataFrame({"uid": ["u1"], "zarr_url": [str(z)], "study_id": ["S1"]})
    fc = SpatialDataCohort(df, filter={"organism": "human"}).freeze(workers=1)
    assert fc.members.loc[0, "fingerprint"].startswith("sha256:")
    assert fc.hash == cohort_hash({"organism": "human"}, fc.members)
    again = SpatialDataCohort(df, filter={"organism": "human"}).freeze(workers=1)
    assert again.hash == fc.hash
