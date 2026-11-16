"""Check that 2 x 2 sums of 8 µm Visium HD bins reproduce Space Ranger's 16 µm table (Task 5 step 0).

Run on a store that holds both tables. It reports the bin counts, the share of 16 µm bins matched by grid
position, the share whose summed counts equal Space Ranger's exactly, and the correlation of per-bin totals.

    python scripts/hd_aggregate_check.py <uid> --release 2026-10-28
"""

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    """Load both tables of one store and compare them bin by bin."""
    from primers.cohort import HD_FINE_TABLE, HD_TABLE, aggregate_bins

    import sddb
    from sddb.cohort import SpatialDataCohort

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("uid")
    ap.add_argument("--release", required=True)
    args = ap.parse_args()
    row = sddb.Catalog(version=args.release).cohort(uids=[args.uid]).to_df()
    fine = SpatialDataCohort(row).to_anndata(table=HD_FINE_TABLE, max_workers=1)
    coarse = SpatialDataCohort(row).to_anndata(table=HD_TABLE, max_workers=1)
    summed = aggregate_bins(fine)
    genes = coarse.var_names.intersection(summed.var_names)
    key = ["array_row", "array_col"]
    a = summed.obs[key].reset_index(drop=True).assign(i=np.arange(summed.n_obs))
    b = coarse.obs[key].astype(int).reset_index(drop=True).assign(j=np.arange(coarse.n_obs))
    m = b.merge(a, on=key, how="left")
    matched = m["i"].notna()
    i, j = m.loc[matched, "i"].astype(int).to_numpy(), m.loc[matched, "j"].to_numpy()
    xs = summed[:, genes].X[i]
    xc = coarse[:, genes].X[j]
    ts, tc = np.asarray(xs.sum(axis=1)).ravel(), np.asarray(xc.sum(axis=1)).ravel()
    exact = np.asarray((abs(xs - xc)).sum(axis=1)).ravel() == 0
    print(
        pd.Series(
            {
                "bins_8um": fine.n_obs,
                "bins_summed": summed.n_obs,
                "bins_16um": coarse.n_obs,
                "share_16um_matched": round(float(matched.mean()), 4),
                "share_matched_exact": round(float(exact.mean()), 4),
                "total_correlation": round(float(np.corrcoef(ts, tc)[0, 1]), 4),
                "total_ratio_summed_over_16um": round(float(ts.sum() / tc.sum()), 4),
                "genes_compared": len(genes),
            }
        ).to_string()
    )


if __name__ == "__main__":
    main()
