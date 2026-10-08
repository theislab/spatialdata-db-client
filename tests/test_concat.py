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
    adata = concat_tables(members, verify=False)
    assert adata.n_obs == 10
    assert set(adata.obs["uid"]) == {"a", "b"}


def test_concat_missing_table_names_member(tmp_path):
    a = _store(tmp_path / "a.zarr", 4)
    members = pd.DataFrame({"uid": ["a"], "zarr_url": [a], "fingerprint": ["x"], "default_table": ["nope"]})
    with pytest.raises(KeyError) as ei:
        concat_tables(members, verify=False)
    assert "'a'" in str(ei.value)


def test_concat_without_default_table_column(tmp_path):
    a = _store(tmp_path / "a.zarr", 2)
    members = pd.DataFrame({"uid": ["a"], "zarr_url": [a]})
    assert concat_tables(members, verify=False).n_obs == 2


def test_cohort_delegators(tmp_path):
    from sddb.cohort import SpatialDataCohort
    from sddb.dataset import Source
    from sddb.freeze import FrozenCohort

    a = _store(tmp_path / "a.zarr", 3)
    df = pd.DataFrame({"uid": ["a"], "zarr_url": [a], "fingerprint": ["x"]})
    assert SpatialDataCohort(df).to_anndata().n_obs == 3
    fc = FrozenCohort("1", "now", Source(), {}, "sha256:h", df)
    assert fc.to_anndata(verify=False).n_obs == 3


@pytest.mark.network
def test_concat_remote_catalog_objects():
    from sddb import Catalog

    cohort = Catalog().query()
    df = cohort.to_df().head(2)
    assert len(df) == 2
    adata = cohort[:2].to_anndata()
    assert adata.n_obs > 0
    assert set(adata.obs["uid"]) == set(df["uid"].astype(str))
