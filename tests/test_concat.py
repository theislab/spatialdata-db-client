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


class _MouseOnly:
    """Skips non-mouse rows cheaply, tables without var names authoritatively, keeps one gene."""

    name = "mouse-only"
    version = "1"

    def __init__(self, keep=True):
        self.keep = keep

    def requirements(self):
        return {"elements": ("X", "obs", "var")}

    def validate_meta(self, row):
        return [] if row["organism"] == "mouse" else [f"organism is {row['organism']}"]

    def validate(self, adata, row):
        return [] if adata.n_vars else ["no genes"]

    def build(self, adata, row):
        return adata[:, :1].copy() if self.keep else None


def test_concat_adapter_maps_and_reports_skips(tmp_path):
    members = _members(tmp_path, [2, 3, 4])
    members.loc[0, "zarr_url"] = str(tmp_path / "missing.zarr")  # skipped by validate_meta, so never read
    adata = concat_tables(members, adapter=_MouseOnly())
    assert adata.shape == (3, 1)
    assert set(adata.obs["uid"]) == {"u1"}
    assert adata.uns["sddb"] == {
        "adapter": "mouse-only",
        "adapter_version": "1",
        "dropped": {"u0": "organism is human", "u2": "organism is human"},
    }


def test_concat_adapter_build_none_is_reported(tmp_path):
    members = _members(tmp_path, [2, 3])
    members["organism"] = "mouse"
    with pytest.raises(ValueError, match="dropped by mouse-only.build"):
        concat_tables(members, adapter=_MouseOnly(keep=False))


def test_concat_adapter_requirements_select_elements(tmp_path):
    path = tmp_path / "l.zarr"
    a = ad.AnnData(X=np.ones((2, 3), "float32"), layers={"counts": np.ones((2, 3), "float32")})
    SpatialData(tables={"table": a}).write(path)
    members = pd.DataFrame({"uid": ["l"], "zarr_url": [str(path)], "organism": ["mouse"]})
    assert "counts" not in concat_tables(members, adapter=_MouseOnly()).layers


def test_table_adapter_protocol():
    from sddb import TableAdapter

    assert isinstance(_MouseOnly(), TableAdapter)
