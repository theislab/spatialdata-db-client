import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def _mod():
    """Import scripts/hd_bin_size.py, which is not part of the package."""
    spec = importlib.util.spec_from_file_location(
        "hd_bin_size", Path(__file__).resolve().parents[1] / "scripts" / "hd_bin_size.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_marker_signal_counts_clusters_and_t_cells():
    """A cluster counts when a marker is twice its background mean and detected in at least 10% of bins."""
    X = np.zeros((40, 3), "float32")
    X[:10, 0] = 3.0  # cluster a: EPCAM high
    X[10:20, 1] = 2.0  # cluster b: CD3E high
    X[20:40, 2] = 1.0  # clusters c, d: ACTB everywhere there, no marker
    adata = ad.AnnData(X, var=pd.DataFrame(index=["EPCAM", "CD3E", "ACTB"]))
    adata.obs["leiden"] = ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10
    share, t_cell = _mod().marker_signal(adata, {"epithelial": ["EPCAM"], "t_cell": ["Cd3e"], "none": ["NOPE"]})
    assert share == 0.5
    assert t_cell
