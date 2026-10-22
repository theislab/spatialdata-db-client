from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData

from sddb.concat import concat_tables


def _store(path, n):
    SpatialData(tables={"table": ad.AnnData(X=np.ones((n, 3), "float32"))}).write(path)
    return str(path)


def test_concat_reads_only_tables(tmp_path):
    a = _store(tmp_path / "a.zarr", 4)
    b = _store(tmp_path / "b.zarr", 6)
    members = pd.DataFrame(
        {"uid": ["a", "b"], "zarr_url": [a, b], "fingerprint": ["x", "y"], "default_table": ["table", "table"]}
    )
    adata = concat_tables(members)
    assert adata.n_obs == 10
    assert set(adata.obs["uid"]) == {"a", "b"}


def test_concat_missing_table_names_member(tmp_path):
    a = _store(tmp_path / "a.zarr", 4)
    members = pd.DataFrame({"uid": ["a"], "zarr_url": [a], "fingerprint": ["x"], "default_table": ["nope"]})
    with pytest.raises(KeyError) as ei:
        concat_tables(members)
    assert "'a'" in str(ei.value)


def test_concat_without_default_table_column(tmp_path):
    a = _store(tmp_path / "a.zarr", 2)
    members = pd.DataFrame({"uid": ["a"], "zarr_url": [a]})
    assert concat_tables(members).n_obs == 2


def test_cohort_delegators(tmp_path):
    from sddb.cohort import SpatialDataCohort

    a = _store(tmp_path / "a.zarr", 3)
    df = pd.DataFrame({"uid": ["a"], "zarr_url": [a], "fingerprint": ["x"]})
    assert SpatialDataCohort(df).to_anndata().n_obs == 3


@pytest.mark.network
def test_concat_remote_catalog_objects():
    from sddb import Catalog

    cohort = Catalog().query()
    df = cohort.to_df().head(2)
    assert len(df) == 2
    adata = cohort[:2].to_anndata()
    assert adata.n_obs > 0
    assert set(adata.obs["uid"]) == set(df["uid"].astype(str))


def _members(tmp_path, sizes):
    urls = [_store(tmp_path / f"{i}.zarr", n) for i, n in enumerate(sizes)]
    return pd.DataFrame(
        {
            "uid": [f"u{i}" for i in range(len(sizes))],
            "zarr_url": urls,
            "organism": ["human", "mouse", "human"][: len(sizes)],
        }
    )


def test_concat_obs_columns_follow_members_under_parallel_reads(tmp_path):
    members = _members(tmp_path, [2, 3, 4])
    adata = concat_tables(members, obs_columns=["organism"], max_workers=3)
    by_uid = adata.obs.groupby("uid", observed=True)["organism"].first().to_dict()
    assert by_uid == {"u0": "human", "u1": "mouse", "u2": "human"}
    assert adata.obs["uid"].value_counts().to_dict() == {"u2": 4, "u1": 3, "u0": 2}
    assert adata.obs["organism"].dtype == "category"


def test_concat_unknown_obs_column_raises(tmp_path):
    with pytest.raises(KeyError, match="tissue"):
        concat_tables(_members(tmp_path, [2]), obs_columns=["tissue"])


def test_concat_elements_skips_layers(tmp_path):
    path = tmp_path / "l.zarr"
    a = ad.AnnData(X=np.ones((2, 3), "float32"), layers={"counts": np.ones((2, 3), "float32")})
    SpatialData(tables={"table": a}).write(path)
    members = pd.DataFrame({"uid": ["l"], "zarr_url": [str(path)]})
    assert "counts" in concat_tables(members).layers
    assert "counts" not in concat_tables(members, elements=("X", "obs", "var")).layers


def test_concat_transform_subsets_and_drops(tmp_path):
    members = _members(tmp_path, [2, 3])

    def keep_mouse_first_gene(adata, row):
        return adata[:, :1].copy() if row["organism"] == "mouse" else None

    adata = concat_tables(members, transform=keep_mouse_first_gene)
    assert adata.shape == (3, 1)
    assert set(adata.obs["uid"]) == {"u1"}
