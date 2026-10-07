from __future__ import annotations

import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

from sddb.cohort import SpatialDataCohort
from sddb.freeze import CohortDriftError, verify_members


def _write(path, n_obs):
    img = Image2DModel.parse(np.zeros((3, 8, 8), "uint8"), dims=("c", "y", "x"))
    SpatialData(
        images={"img": img},
        tables={"table": ad.AnnData(X=np.ones((n_obs, 3), "float32"))},
    ).write(path)
    return path


def test_verify_passes_then_detects_drift(tmp_path):
    z = _write(tmp_path / "o.zarr", 4)
    df = pd.DataFrame({"uid": ["o"], "zarr_url": [str(z)], "study_id": ["S"]})
    fc = SpatialDataCohort(df, filter={}).freeze()
    verify_members(fc.members)  # no raise: unchanged
    shutil.rmtree(z)
    _write(z, 9)  # republish different data at the same path
    with pytest.raises(CohortDriftError) as ei:
        verify_members(fc.members)
    assert "o" in ei.value.drifted
