import pandas as pd

from sddb.cohort import SpatialDataCohort
from sddb.dataset import Source
from sddb.manifest import CohortManifest


def _cohort(n=3):
    df = pd.DataFrame(
        {
            "uid": [f"{'u' * 12}{i:04d}0000" for i in range(n)],
            "zarr_url": [f"s3://b/.lamindb/u{i}.zarr" for i in range(n)],
            "study_id": ["s"] * n,
            "technology": ["Visium HD"] * n,
        }
    )
    return SpatialDataCohort(df, source=Source("cat.parquet", version="v1"), filter={"organism": "human"})


def test_freeze_is_offline_and_roundtrips(tmp_path):
    coh = _cohort(3)
    m = coh.freeze()
    assert isinstance(m, CohortManifest)
    assert m.filter == {"organism": "human"}
    p = m.write(tmp_path / "c.json")
    coh2 = SpatialDataCohort.from_manifest(CohortManifest.load(p))
    assert coh2.freeze().hash == m.hash
    assert list(coh2.to_df()["uid"]) == list(coh.to_df()["uid"])


def test_freeze_hash_tracks_membership():
    assert _cohort(3).freeze().hash != _cohort(2).freeze().hash
