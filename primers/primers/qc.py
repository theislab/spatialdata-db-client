"""Per-technology QC for a cohort family (Task 5).

Thresholds come from ``thresholds.yaml``, grouped by technology (spot, binned spot, imaging cell). Control
probes are left out before the metrics are computed. The mitochondrial share is computed only when the
table has ``MT-`` genes (whole-transcriptome assays); targeted panels have none.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from primers.markers import data_path, panel_genes

THRESHOLDS = data_path("thresholds.yaml")
MIN_GENES_FOR_QC = 50
KEPT, NO_CELLS = "kept", "dropped: no cells after QC"


def load_thresholds(path: str | Path = THRESHOLDS) -> dict[str, dict]:
    """The threshold groups by name, each with ``technologies``, filters, ``flag`` bounds and ``reason``."""
    return yaml.safe_load(Path(path).read_text())


def _group_of(thresholds: dict[str, dict]) -> dict[str, dict]:
    return {tech: group for group in thresholds.values() for tech in group["technologies"]}


def _metrics(adata: ad.AnnData) -> pd.DataFrame:
    """total_counts, n_genes (integers) and pct_mt per observation (pct_mt is NaN without MT genes)."""
    import scanpy as sc

    mt = adata.var_names.str.upper().str.startswith("MT-")
    tmp = ad.AnnData(
        X=adata.X, obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame({"mt": mt}, index=adata.var_names)
    )
    qc, _ = sc.pp.calculate_qc_metrics(
        tmp, qc_vars=["mt"] if mt.any() else [], percent_top=None, log1p=False, inplace=False
    )
    return pd.DataFrame(
        {
            "total_counts": np.rint(qc["total_counts"].to_numpy()).astype(np.int64),
            "n_genes": qc["n_genes_by_counts"].to_numpy().astype(np.int64),
            "pct_mt": qc["pct_counts_mt"].to_numpy(dtype=float) if mt.any() else np.full(adata.n_obs, np.nan),
        },
        index=adata.obs_names,
    )


def apply_qc(adata: ad.AnnData, thresholds: dict[str, dict]) -> tuple[ad.AnnData, pd.DataFrame]:
    """Filter observations per technology and describe each dataset.

    ``adata.obs`` needs ``dataset_uid`` and ``technology``. Returns the kept observations, with
    ``total_counts``, ``n_genes`` and ``pct_mt`` in obs, and one QC row per dataset: ``dataset_uid``,
    ``technology``, ``n_before``, ``n_after``, ``median_counts``, ``median_genes``, ``pct_mt_median`` (NaN
    without MT genes), ``flagged``, ``flag_reason`` and ``status`` (``kept`` or ``dropped: no cells after
    QC``). Medians are over all observations before filtering. Raises ``ValueError`` with fewer than 50
    genes, or for a technology with no threshold group.
    """
    adata = adata[:, np.isin(adata.var_names, panel_genes(adata.var_names))]
    if adata.n_vars < MIN_GENES_FOR_QC:
        raise ValueError(f"QC needs at least {MIN_GENES_FOR_QC} genes; got {adata.n_vars} genes")
    groups = _group_of(thresholds)
    if missing := set(adata.obs["technology"].astype(str)) - set(groups):
        raise ValueError(f"no QC thresholds for technology {sorted(missing)}")
    m = _metrics(adata)
    tech = adata.obs["technology"].astype(str)
    g = tech.map(groups)
    keep = (m["total_counts"] >= g.map(lambda x: x["min_counts"])) & (m["n_genes"] >= g.map(lambda x: x["min_genes"]))
    cap = g.map(lambda x: x.get("max_pct_mt", np.inf))
    keep &= m["pct_mt"].isna() | (m["pct_mt"] <= cap)

    rows = []
    for (uid, t), idx in adata.obs.groupby(["dataset_uid", "technology"], observed=True).groups.items():
        mm, bounds = m.loc[idx], groups[str(t)]["flag"]
        n_before, n_after = len(idx), int(keep.loc[idx].sum())
        reasons = []
        if (med := float(mm["total_counts"].median())) < bounds["min_median_counts"]:
            reasons.append(f"median counts {med:g} < {bounds['min_median_counts']}")
        if (share := n_after / n_before) < bounds["min_share_kept"]:
            reasons.append(f"{share:.0%} kept < {bounds['min_share_kept']:.0%}")
        rows.append(
            {
                "dataset_uid": str(uid),
                "technology": str(t),
                "n_before": n_before,
                "n_after": n_after,
                "median_counts": med,
                "median_genes": float(mm["n_genes"].median()),
                "pct_mt_median": float(mm["pct_mt"].median()) if mm["pct_mt"].notna().any() else np.nan,
                "flagged": bool(reasons),
                "flag_reason": "; ".join(reasons),
                "status": KEPT if n_after else NO_CELLS,
            }
        )
    out = adata[keep.to_numpy()].copy()
    for col in ("total_counts", "n_genes", "pct_mt"):
        out.obs[col] = m.loc[keep, col].to_numpy()
    return out, pd.DataFrame(rows)


def floor_of(adata: ad.AnnData, floor: float | Mapping[str, float]) -> np.ndarray:
    """The count floor of each observation: ``floor`` itself, or ``floor[technology]`` for a mapping."""
    if isinstance(floor, Mapping):
        return adata.obs["technology"].astype(str).map(floor).to_numpy(dtype=float)
    return np.full(adata.n_obs, float(floor))


def floor_loss(
    adata: ad.AnnData, floor: float | Mapping[str, float], genes: list[str] | None = None
) -> dict[str, float]:
    """Per technology, the share of observations with fewer than ``floor`` counts (on ``genes`` if given).

    ``floor`` is one number or a per-technology mapping (each technology's QC ``min_counts``).

    Gate 3 of the joint view uses it on the shared genes: a technology that loses more than half its
    observations is left out of the joint view.
    """
    sub = adata if genes is None else adata[:, np.isin(adata.var_names, genes)]
    total = np.asarray(sub.X.sum(axis=1)).ravel()
    low = pd.Series(total < floor_of(adata, floor), index=adata.obs_names)
    return {str(t): float(low[idx].mean()) for t, idx in adata.obs.groupby("technology", observed=True).groups.items()}


def qc_floors(thresholds: dict[str, dict]) -> dict[str, float]:
    """Each technology's count floor (``min_counts`` of its threshold group), for gate 3 and the joint view."""
    return {t: float(g["min_counts"]) for t, g in _group_of(thresholds).items()}
