"""Choose the Visium HD table (8 or 16 µm bins) and the count floor for primers (Task 5, step 0).

For each Visium HD section of a collection and each (table, floor) option, the script reports:

- ``share_kept``: the share of bins with at least ``floor`` counts;
- ``share_clusters_with_marker``: after per-section PCA, neighbours and Leiden on the kept bins, the share
  of clusters in which at least one compartment marker (markers.yaml) is above background. A marker is
  above background in a cluster when its mean log-normalised expression there is at least twice its
  mean in the other bins and at least 10% of the cluster's bins detect it;
- ``t_cell_cluster``: whether any cluster shows a T-cell marker above background (Oliveira 2025 found
  that 8 µm bins turn small cells such as T cells into doublets or rejects).

Runs on a GPU with rapids-singlecell. Usage:

    python scripts/hd_bin_size.py oliveira-2024-colorectal colon --release 2026-10-28 --out hd_bin_size.tsv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from primers.markers import load_markers

OPTIONS = {"square_008um": (50, 100, 200), "square_016um": (100, 200, 400)}
LEIDEN_RESOLUTION = 1.0
FOLD, MIN_DETECTED = 2.0, 0.1


def marker_signal(adata, compartments: dict[str, list[str]], cluster_key: str = "leiden") -> tuple[float, bool]:
    """Share of clusters with a compartment marker above background, and whether a T-cell cluster exists."""
    import scipy.sparse as sp

    upper = {g.upper(): g for g in adata.var_names}
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    with_marker, t_cell = 0, False
    labels = np.unique(clusters)
    for c in labels:
        inside = clusters == c
        hit = False
        for name, genes in compartments.items():
            cols = [upper[g.upper()] for g in genes if g.upper() in upper]
            if not cols:
                continue
            X = adata[:, cols].X
            X = X.toarray() if sp.issparse(X) else np.asarray(X)
            mean_in, mean_out = X[inside].mean(0), X[~inside].mean(0)
            detected = (X[inside] > 0).mean(0)
            ok = (mean_in >= FOLD * np.maximum(mean_out, 1e-9)) & (detected >= MIN_DETECTED)
            if ok.any():
                hit = True
                t_cell |= name == "t_cell"
        with_marker += hit
    return with_marker / len(labels), t_cell


def run_option(adata, floor: int, compartments: dict) -> dict:
    """Filter at ``floor``, cluster on the GPU, and score marker signal."""
    import rapids_singlecell as rsc
    import scanpy as sc

    n = adata.n_obs
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
    kept = adata[adata.obs["total_counts"] >= floor].copy()
    kept = kept[:, np.asarray(kept.X.sum(axis=0)).ravel() > 0].copy()  # the GPU sparse PCA refuses unexpressed genes
    out = {"n_bins": n, "n_kept": kept.n_obs, "share_kept": round(kept.n_obs / n, 4)}
    if kept.n_obs < 100:
        return {**out, "n_clusters": 0, "share_clusters_with_marker": np.nan, "t_cell_cluster": False}
    rsc.get.anndata_to_GPU(kept)
    rsc.pp.normalize_total(kept)
    rsc.pp.log1p(kept)
    rsc.pp.pca(kept, n_comps=50)
    rsc.pp.neighbors(kept)
    rsc.tl.leiden(kept, resolution=LEIDEN_RESOLUTION, random_state=0)
    rsc.get.anndata_to_CPU(kept)
    share, t_cell = marker_signal(kept, compartments)
    return {
        **out,
        "n_clusters": kept.obs["leiden"].nunique(),
        "share_clusters_with_marker": round(share, 4),
        "t_cell_cluster": t_cell,
    }


def main() -> None:
    """Run every option on every Visium HD section of the collection and write one row per option."""
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    import sddb
    from sddb.cohort import SpatialDataCohort

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("collection")
    ap.add_argument("marker_set")
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rmm.reinitialize(managed_memory=True)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    compartments = load_markers()[args.marker_set]["compartments"]
    members = sddb.Catalog(version=args.release).collection(args.collection).to_df()
    members = members[members["technology"] == "VisiumHD"]
    rows = []
    for _, m in members.iterrows():
        for table, floors in OPTIONS.items():
            adata = SpatialDataCohort(members[members["uid"] == m["uid"]]).to_anndata(
                table=table, gene_key="symbol", max_workers=1
            )
            for floor in floors:
                res = run_option(adata.copy(), floor, compartments)
                rows.append({"uid": m["uid"], "title": m["title"], "table": table, "floor": floor, **res})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
