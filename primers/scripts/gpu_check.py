"""Check the GPU environment and run the GPU smoke test.

Prints the GPU and library versions, runs PCA, Harmony and neighbours on a small random matrix, then runs
``primers.embed.embed_views`` on the GPU on a small synthetic two-family cohort.

Used to check the environment on each GPU type before any primer run (Task 2, step 1).
"""

from importlib.metadata import version

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

# the configuration the primer runs use: managed memory, so a cohort may spill past GPU memory
rmm.reinitialize(managed_memory=True)
cp.cuda.set_allocator(rmm_cupy_allocator)

props = cp.cuda.runtime.getDeviceProperties(0)
print("gpu", props["name"].decode(), "memory_gb", round(props["totalGlobalMem"] / 1e9, 1))
for dist in (
    "spatialdata-db",
    "rapids-singlecell-cu13",
    "harmonypy",
    "scanpy",
    "anndata",
    "spatialdata",
    "spatialdata-plot",
    "cupy",
    "rmm",
    "cuml",
    "cugraph",
):
    try:
        print(dist, version(dist))
    except Exception:  # not installed under this distribution name
        print(dist, "not installed")
rng = np.random.default_rng(0)
adata = ad.AnnData(rng.poisson(1.0, (20000, 300)).astype("float32"))
adata.obs["batch"] = rng.choice(["a", "b", "c"], adata.n_obs)
rsc.get.anndata_to_GPU(adata)
rsc.pp.normalize_total(adata)
rsc.pp.log1p(adata)
rsc.pp.pca(adata, n_comps=30)
rsc.pp.harmony_integrate(adata, key="batch")
rsc.pp.neighbors(adata, use_rep="X_pca_harmony")
rsc.tl.umap(adata)
print("ok", adata.obsm["X_umap"].shape)

# GPU smoke test of the embedding: 3 sections x 2,000 observations per family, 2,500 spot genes and
# 300 imaging genes, 220 of them shared
from primers.embed import embed_views, mixing_stats  # noqa: E402

shared = ["EPCAM", "PTPRC"] + [f"S{i}" for i in range(218)]
panels = {
    "spot": ("Visium", shared + [f"V{i}" for i in range(2280)], 3000.0),
    "imaging_cell": ("Xenium", shared + [f"X{i}" for i in range(80)], 150.0),
}
cohort = {}
for fam, (tech, genes, depth) in panels.items():
    parts = []
    for s in range(3):
        uid = f"{fam}-{s}"
        obs = pd.DataFrame(
            {"dataset_uid": uid, "technology": tech, "paper": f"paper-{uid}"},
            index=[f"{uid}-{i}" for i in range(2000)],
        )
        X = rng.poisson(depth / len(genes), (2000, len(genes))).astype("float32")
        parts.append(ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes)))
    cohort[fam] = ad.concat(parts)
    cohort[fam].uns["member_genes"] = {f"{fam}-{s}": genes for s in range(3)}
markers = {"compartments": {"epithelial": ["EPCAM"], "immune": ["PTPRC"]}}
views = embed_views(cohort, markers, floor=10, use_gpu=True)
for name, v in views.items():
    print("view", name, v.shape, {k: val for k, val in v.uns["primer"].items() if k != "gates"})
    for basis in ("X_pca", "X_pca_harmony"):
        if basis in v.obsm:
            m = mixing_stats(v, "dataset_uid", basis)
            print(" ", basis, "entropy", m["entropy"]["mean"], "other_share", m["other_share"]["mean"])
print("gates", next(iter(views.values())).uns["primer"].get("gates"))
