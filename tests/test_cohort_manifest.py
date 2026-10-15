import pandas as pd

from sddb.dataset import Source
from sddb.manifest import SCHEMA_VERSION, CohortManifest, cohort_hash


def _members(n=3):
    return pd.DataFrame(
        {
            "uid": [f"uid{i:016d}0000"[-20:] for i in range(n)],
            "zarr_url": [f"s3://b/.lamindb/uid{i}.zarr" for i in range(n)],
            "study_id": ["s"] * n,
            "database_version": ["0.0.14"] * n,
        }
    )


def test_hash_is_order_independent_and_version_sensitive():
    m = _members(3)
    h1 = cohort_hash({"organism": "human"}, m["uid"])
    h2 = cohort_hash({"organism": "human"}, m["uid"][::-1])  # reversed order
    assert h1 == h2
    assert h1.startswith("sha256:")
    changed = m["uid"].tolist()
    changed[0] = changed[0][:16] + "0001"  # same stem, newer version
    assert cohort_hash({"organism": "human"}, pd.Series(changed)) != h1


def test_manifest_roundtrip_small(tmp_path):
    m = _members(3)
    man = CohortManifest(
        SCHEMA_VERSION,
        "2026-10-15T00:00:00+00:00",
        Source("cat.parquet", version="v1"),
        {"organism": "human"},
        cohort_hash({"organism": "human"}, m["uid"]),
        m,
    )
    p = man.write(tmp_path / "cohort.json")
    back = CohortManifest.load(p)
    assert back.hash == man.hash
    assert back.filter == {"organism": "human"}
    assert sorted(back.members["uid"]) == sorted(m["uid"])
    assert back.source.version == "v1"


def test_manifest_roundtrip_large_uses_parquet_sidecar(tmp_path):
    m = _members(250)  # over the 200 sidecar threshold
    man = CohortManifest(
        SCHEMA_VERSION,
        "2026-10-15T00:00:00+00:00",
        Source("cat.parquet"),
        {},
        cohort_hash({}, m["uid"]),
        m,
    )
    p = man.write(tmp_path / "big.json")
    assert (tmp_path / "big.members.parquet").exists()
    back = CohortManifest.load(p)
    assert len(back.members) == 250
    assert back.hash == man.hash
