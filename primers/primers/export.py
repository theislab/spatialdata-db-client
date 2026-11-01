"""Write a primer's bundle (Task 6): manifest.json, umap.parquet, qc.parquet, cohort.parquet, embedding.h5ad.

See README.md, "Bundle contract". The notebook and its HTML are added by the runner (Task 7).
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from primers.cohort import MIN_PAPERS
from primers.embed import N_PCS, SEED, _host, mixing_stats
from primers.qc import NO_CELLS

WEB_CAP = 50_000
MAX_H5AD_BYTES = 2 * 1024**3  # GitHub's release-asset limit
EMBEDDING_TARGET = 1024**3  # per embedding file; a cohort over it is split into parts of whole datasets
UMAP_COLUMNS = ["dataset_uid", "paper", "technology", "tissue", "total_counts", "n_genes"]
EMBEDDINGS = ("X_pca", "X_pca_harmony", "X_umap", "X_umap_harmony")
DISTRIBUTIONS = (
    "spatialdata-db",
    "scanpy",
    "anndata",
    "spatialdata",
    "spatialdata-plot",
    "rapids-singlecell-cu13",
    "rapids-singlecell",
    "harmonypy",
)


def equal_share_subsample(uids: pd.Series, cap: int = WEB_CAP, seed: int = SEED) -> np.ndarray:
    """Positions of at most ``cap`` observations with an equal share per dataset.

    A dataset smaller than its share is kept whole, and its unused share is split among the others.
    """
    sizes = uids.value_counts()
    quota, left, budget = {}, sizes.copy(), cap
    while len(left):
        share = budget // len(left)
        small = left[left <= share]
        if small.empty:
            quota.update(dict.fromkeys(left.index, share))
            break
        quota.update(small.to_dict())
        budget -= int(small.sum())
        left = left.drop(small.index)
    rng = np.random.default_rng(seed)
    where = uids.groupby(uids.to_numpy()).indices
    picked = [rng.choice(where[uid], size=n, replace=False) for uid, n in quota.items()]
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)


def _versions() -> dict[str, str]:
    out = {}
    for dist in DISTRIBUTIONS:
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            continue
    return out


def _hardware(path: str) -> str:
    if path == "gpu":
        import cupy as cp

        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return f"{platform.machine()} CPU"


def _pad(x: np.ndarray, width: int) -> np.ndarray:
    """``x`` as float16 with exactly ``width`` columns (NaN padding when a view has fewer components)."""
    out = np.full((x.shape[0], width), np.nan, dtype=np.float16)
    out[:, : min(width, x.shape[1])] = x[:, :width]
    return out


def _subsamples(views: dict[str, ad.AnnData]) -> dict[str, np.ndarray]:
    """Per view, the positions of the observations that go into umap.parquet."""
    return {
        name: equal_share_subsample(v.obs["dataset_uid"].astype(str).reset_index(drop=True))
        for name, v in views.items()
    }


def _embedding(views: dict[str, ad.AnnData], picked: dict[str, np.ndarray]) -> ad.AnnData:
    """Each observation once (from its family view), with the family and joint embeddings in obsm.

    obs gets one boolean ``in_web_subsample_<view>`` column per view (``:`` read as ``_``), marking the
    observations of that view in umap.parquet.
    """
    families = [v for k, v in views.items() if k != "joint"]
    obs = pd.concat([v.obs.assign(view=k) for k, v in views.items() if k != "joint"])
    for name, v in views.items():
        chosen = set(v.obs_names[picked[name]])
        obs[f"in_web_subsample_{name.replace(':', '_')}"] = obs.index.isin(chosen) & (
            (obs["view"] == name).to_numpy() if name != "joint" else True
        )
    obsm = {}
    for key in EMBEDDINGS:
        width = N_PCS if key.startswith("X_pca") else 2
        blocks = [
            _pad(_host(v.obsm[key]), width) if key in v.obsm else np.full((v.n_obs, width), np.nan, np.float16)
            for v in families
        ]
        obsm[key] = np.vstack(blocks)
    if "joint" in views:
        j = views["joint"]
        for key in EMBEDDINGS:
            if key not in j.obsm:
                continue
            width = N_PCS if key.startswith("X_pca") else 2
            frame = pd.DataFrame(_pad(_host(j.obsm[key]), width).astype(np.float32), index=j.obs_names)
            obsm[f"{key}_joint"] = frame.reindex(obs.index).to_numpy(dtype=np.float16)
    return ad.AnnData(obs=obs, obsm=obsm)


def _part(path: Path, adata: ad.AnnData, uids: list[str]) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "n_obs": int(adata.n_obs), "dataset_uids": uids}


def _write_embedding(emb: ad.AnnData, out: Path, target: int) -> list[dict]:
    """Write ``emb`` as embedding.h5ad, or as embedding-1.h5ad, embedding-2.h5ad, ... when it is over ``target``.

    A part holds whole datasets, in their order in ``emb``, so each part reads on its own and
    ``anndata.concat`` over the parts in order gives ``emb`` back. Returns the parts as ``{file, bytes,
    n_obs, dataset_uids}``. Raises ``ValueError`` when one dataset alone is over ``target``.
    """
    uids = emb.obs["dataset_uid"].astype(str)
    order = list(dict.fromkeys(uids))
    emb = emb[np.argsort(uids.map({u: i for i, u in enumerate(order)}).to_numpy(), kind="stable")]
    uids = emb.obs["dataset_uid"].astype(str)
    one = out / "embedding.h5ad"
    emb.write_h5ad(one, compression="gzip")
    size = one.stat().st_size
    if size <= target:
        return [_part(one, emb, order)]
    one.unlink()
    counts = uids.value_counts()
    budget = int(0.9 * target * emb.n_obs / size)  # observations per part, from the bytes per observation
    while True:
        groups, n = [[]], 0
        for u in order:
            if n + counts[u] > budget and groups[-1]:
                groups.append([])
                n = 0
            groups[-1].append(u)
            n += counts[u]
        parts = []
        for i, group in enumerate(groups, 1):
            sub = emb[uids.isin(group).to_numpy()]
            sub.write_h5ad(path := out / f"embedding-{i}.h5ad", compression="gzip")
            parts.append(_part(path, sub, group))
        over = [p for p in parts if p["bytes"] > target]
        if not over:
            return parts
        for p in parts:
            (out / p["file"]).unlink()
        if alone := next((p for p in over if len(p["dataset_uids"]) == 1), None):
            raise ValueError(f"dataset {alone['dataset_uids'][0]} alone is over the {target}-byte embedding target")
        budget = int(budget * 0.8)  # ponytail: shrink and rewrite; parts compress unevenly


def _umap_table(views: dict[str, ad.AnnData], picked: dict[str, np.ndarray]) -> pd.DataFrame:
    """The web subsample of each view; missing labels stay null and the counts stay integers."""
    frames = []
    for name, v in views.items():
        idx = picked[name]
        sub = v.obs.iloc[idx].reindex(columns=UMAP_COLUMNS).reset_index(drop=True)
        u = _host(v.obsm["X_umap"])[idx]
        h = _host(v.obsm["X_umap_harmony"])[idx] if "X_umap_harmony" in v.obsm else np.full((len(idx), 2), np.nan)
        frames.append(sub.assign(view=name, x=u[:, 0], y=u[:, 1], x_harmony=h[:, 0], y_harmony=h[:, 1]))
    out = pd.concat(frames, ignore_index=True)
    for col in ("dataset_uid", "paper", "technology", "tissue", "view"):
        out[col] = pd.Series([None if pd.isna(x) else str(x) for x in out[col].astype(object)], dtype=object)
    for col in ("total_counts", "n_genes"):
        out[col] = pd.to_numeric(out[col]).round().astype("Int64")
    return out[["view", "x", "y", "x_harmony", "y_harmony", *UMAP_COLUMNS]]


def mixing_by_view(views: dict[str, ad.AnnData]) -> dict:
    """Mixing statistics per view, before and after Harmony, as ``{view: {batch: {basis: stats}}}``.

    Per section (serial sections of one 10x block count as one) and per paper, and for the joint view also
    per technology. ``stats`` is ``mixing_stats``.
    """
    out = {}
    for name, v in views.items():
        bases = [b for b in ("X_pca", "X_pca_harmony") if b in v.obsm]
        section = "block" if "block" in v.obs else "dataset_uid"  # serial sections of one 10x block count once
        keys = [("section", section), ("paper", "paper")] + [("technology", "technology")] * (name == "joint")
        out[name] = {key: {b: mixing_stats(v, col, b) for b in bases} for key, col in keys if col in v.obs}
    return out


def with_qc_status(cohort: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    """The cohort table with members that have no cells after QC marked ``dropped: no cells after QC``."""
    cohort = cohort.copy()
    if "status" in qc:
        dropped = set(qc.loc[qc["status"].astype(str) == NO_CELLS, "dataset_uid"].astype(str))
        hit = cohort["uid"].astype(str).isin(dropped) & (cohort["status"].astype(str) == "kept")
        cohort.loc[hit, "status"] = NO_CELLS
    return cohort


def _parquet(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x if x is None or isinstance(x, str) else json.dumps(x, default=str))
    df.to_parquet(path, index=False)


def export_bundle(
    views: dict[str, ad.AnnData],
    cohort: pd.DataFrame,
    qc: pd.DataFrame,
    spec: dict,
    out_dir: str | Path,
    release: str,
    runtime_s: float,
    mixing: dict | None = None,
    primers_ref: str | None = None,
    embedding_target: int = EMBEDDING_TARGET,
) -> Path:
    """Write the bundle for ``spec["slug"]`` under ``out_dir`` and return its directory.

    QC status is merged into the cohort table first: a member with no cells after QC is dropped, and the
    kept counts leave it out. ``mixing`` is ``mixing_by_view(views)`` when the caller has computed it
    already; otherwise it is computed here. ``primers_ref`` is the commit or tag of this code, recorded so a
    bundle names the code that made it. The embedding is one embedding.h5ad, or parts of whole datasets when
    it is over ``embedding_target`` (see ``_write_embedding``); the manifest lists them under ``embedding``.
    Raises ``ValueError`` when fewer than 3 papers remain after QC, or when one dataset alone is over the
    embedding target.
    """
    cohort = with_qc_status(cohort, qc)
    panel = next((v.uns["primer"].get("members_dropped_panel") or {} for v in views.values()), {})
    hit = cohort["uid"].astype(str).isin(panel) & (cohort["status"].astype(str) == "kept")
    cohort.loc[hit, "status"] = cohort.loc[hit, "uid"].astype(str).map(panel)
    status = cohort.set_index("uid")["status"].astype(str)
    kept = cohort[status.to_numpy() == "kept"]
    if (n_papers := kept["paper"].nunique()) < MIN_PAPERS:
        raise ValueError(f"cohort {spec['slug']!r} has {n_papers} papers after QC; at least {MIN_PAPERS} needed")
    out = Path(out_dir) / spec["slug"]
    out.mkdir(parents=True, exist_ok=True)
    picked = _subsamples(views)
    parts = _write_embedding(_embedding(views, picked), out, min(embedding_target, MAX_H5AD_BYTES - 1))
    _parquet(_umap_table(views, picked), out / "umap.parquet")
    _parquet(qc, out / "qc.parquet")
    _parquet(cohort, out / "cohort.parquet")

    inferred = (
        cohort[cohort.get("paper_source", pd.Series(dtype=str)).eq("doi_inferred")]
        if "paper_source" in cohort
        else cohort.iloc[:0]
    )
    path = next(iter(views.values())).uns["primer"]["path"]
    view_meta = {
        name: {
            k: val
            for k, val in v.uns["primer"].items()
            if k not in ("gates", "views_skipped", "floor", "members_dropped_panel")
        }
        for name, v in views.items()
    }
    skipped = next((v.uns["primer"].get("views_skipped", {}) for v in views.values()), {})
    coverage = {
        str(r.uid): {"n_genes": int(r.n_genes), "n_ensembl": int(r.n_ensembl)}
        for r in kept.itertuples()
        if "n_genes" in kept and pd.notna(r.n_genes)
    }
    gates = next((v.uns["primer"]["gates"] for v in views.values() if "gates" in v.uns["primer"]), None)
    technologies = sorted({t for v in views.values() for t in v.obs["technology"].astype(str)})
    manifest = {
        "slug": spec["slug"],
        "title": spec.get("title", spec["slug"]),
        "query": spec.get("query", {}),
        # everything that defines the membership, so the site can tell when the live catalog has moved on
        "spec": {k: spec.get(k) for k in ("query", "include_uids", "exclude_uids", "families")},
        "release": release,
        "primers_ref": primers_ref,
        "members_kept": kept["uid"].astype(str).tolist(),
        "members_dropped": status[status != "kept"].to_dict(),
        "n_datasets": int(len(kept)),
        "n_papers": int(n_papers),
        "technologies": technologies,
        "n_cells_before": int(qc["n_before"].sum()),
        "n_cells_after": int(qc["n_after"].sum()),
        "n_flagged": int(qc["flagged"].sum()),
        "versions": _versions(),
        "seeds": {"global": SEED},
        "hardware": _hardware(path),
        "runtime_s": round(float(runtime_s), 1),
        "path": path,
        "views": view_meta,
        "views_skipped": skipped,
        "floors": next((v.uns["primer"].get("floor") for v in views.values()), None),
        "member_genes": coverage,
        "embedding": parts,
        "embedding_h5ad_bytes": sum(p["bytes"] for p in parts),
        "gates": gates,
        "mixing": mixing_by_view(views) if mixing is None else mixing,
        "paper_inference": dict(
            zip(inferred["uid"].astype(str), inferred["paper_inferred_from"].astype(str), strict=True)
        )
        if len(inferred)
        else {},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    )
    return out
