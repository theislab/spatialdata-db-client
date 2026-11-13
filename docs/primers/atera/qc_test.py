"""T138 preprocessing test on a stratified 100k sample of the six Atera sections (catalog-2026-11-12), before any QC.

Runs the primer's current settings and two proposed variants on the same cells, and writes per-section QC losses and
UMAP grids (uncorrected and Harmony; coloured by section, log1p total counts and cell type) to OUT.

Sources for the proposed rules:
- Lower-tail QC by median absolute deviations (MAD) of log1p total counts and log1p genes per cell, 5 MADs: "we mark
  cells as outliers if they differ by 5 MADs which is a relatively permissive filtering strategy"
  (sc-best-practices.org, Quality Control chapter). Applied per section here because depth differs ~6x between
  sections (qc_per_dataset.csv of job 43189110); the chapter does not say per sample. 3 MADs reported for comparison.
- HVGs: scanpy highly_variable_genes(flavor="seurat_v3", batch_key=section) on counts: "count data is expected"; "genes
  are selected within each batch separately and merged ... This simple process avoids the selection of batch-specific
  genes and acts as a lightweight batch correction method" (scanpy docs, pp.highly_variable_genes).
- Scaling: scanpy pp.scale "Scale data to unit variance and zero mean"; max_value "Clip (truncate) to this value after
  scaling" (scanpy docs, pp.scale).

    pixi run python -I qc_test.py OUT        (GPU node, from docs/primers/atera; SDDB_CACHE_DIR set)
"""
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import sddb  # noqa: E402

CPU = len(sys.argv) > 2 and sys.argv[2] == "cpu"  # scanpy on CPU (Harmony via sc.external.pp.harmony_integrate); default rapids on GPU
if not CPU:
    import rapids_singlecell as rsc

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SEED, PER_SECTION, N_HVG, N_PCS = 0, 16_667, 2_000, 50
MARKERS = {  # as primer.py
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT5", "KRT14", "KRT17"],
    "T/NK": ["CD3E", "CD3D", "CD2", "NKG7", "GZMA"],
    "B/plasma": ["MS4A1", "CD79A", "JCHAIN", "MZB1"],
    "myeloid": ["CD68", "CD14", "LYZ", "CSF1R", "C1QA"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "smooth muscle/pericyte": ["ACTA2", "MYH11", "RGS5", "TAGLN"],
    "endothelial": ["PECAM1", "VWF", "CLDN5", "KDR"],
    "neuron": ["SNAP25", "RBFOX3", "SYT1", "GAD1", "SLC17A7"],
    "astrocyte": ["GFAP", "AQP4", "SLC1A3"],
    "oligodendrocyte": ["MBP", "PLP1", "MOG"],
    "microglia": ["P2RY12", "CX3CR1", "TMEM119"],
}
BRAIN_ONLY = {"neuron", "astrocyte", "oligodendrocyte", "microglia"}

# ---- load: tables only, sampled before QC
atera = sddb.Catalog(version="2026-11-12").query(technology="10x_atera")
m = atera.members
parts = []
rng = np.random.default_rng(SEED)
for r in m.itertuples():
    ds = next(iter(atera.where(m["uid"] == r.uid)))
    a = ds.to_sdata(elements=["tables/table"]).tables["table"]
    ftype = next((c for c in ("feature_types", "feature_type") if c in a.var), None)
    if ftype:
        a = a[:, a.var[ftype].astype(str) == "Gene Expression"]
    if "feature_name" in a.var:
        a.var_names = a.var["feature_name"].astype(str).values
    a.var_names_make_unique()
    idx = np.sort(rng.choice(a.n_obs, size=min(PER_SECTION, a.n_obs), replace=False))
    a = a[idx].copy()
    a.X = a.X.astype(np.float32)
    pres = r.preservation_method if isinstance(r.preservation_method, str) else ""
    a.obs = pd.DataFrame(index=a.obs_names.astype(str) + "-" + r.uid,
                         data={"section": f"{r.tissue} ({'FF' if pres == 'Fresh Frozen' else pres})", "tissue": r.tissue})
    parts.append(a)
    print(r.uid, r.tissue, a.shape, flush=True)
shared = sorted(set.intersection(*(set(p.var_names) for p in parts)))
adata = ad.concat([p[:, shared] for p in parts])
adata.obs["section"] = adata.obs["section"].astype("category")
adata.layers["counts"] = adata.X.copy()
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=True, inplace=True)
print("sample", adata.shape, "shared genes", len(shared), flush=True)


# ---- QC rules
def mad_low(x, k):
    med = np.median(x)
    return x < med - k * np.median(np.abs(x - med))


cur_keep = (adata.obs["total_counts"] >= 50) & (adata.obs["n_genes_by_counts"] >= 20)
low = {}
for k in (5, 3):
    f = pd.Series(False, index=adata.obs_names)
    for s, o in adata.obs.groupby("section", observed=True):
        f[o.index] = mad_low(o["log1p_total_counts"].values, k) | mad_low(o["log1p_n_genes_by_counts"].values, k)
    low[k] = f
prop_keep = cur_keep & ~low[5]
sec_med = adata.obs.groupby("section", observed=True)["total_counts"].transform("median")
lowq = {f: ~cur_keep | low[5] | (adata.obs["total_counts"] < f * sec_med) for f in (0.25, 0.3)}  # our judgement (core_qc.py: the low-count core is small, sparse cells)
for f, v in lowq.items():
    print(f"low quality at {f} x section median:", v.groupby(adata.obs['section'], observed=True).sum().to_dict(), flush=True)
qc = adata.obs.groupby("section", observed=True).agg(sampled=("total_counts", "size"), median_counts=("total_counts", "median"),
                                                     median_genes=("n_genes_by_counts", "median"))
qc["lost_current"] = (~cur_keep).groupby(adata.obs["section"], observed=True).sum()
qc["lost_5mad"] = (~(cur_keep & ~low[5])).groupby(adata.obs["section"], observed=True).sum()
qc["lost_3mad"] = (~(cur_keep & ~low[3])).groupby(adata.obs["section"], observed=True).sum()
for k in (5, 3):  # the thresholds in counts, for the text
    qc[f"min_counts_{k}mad"] = adata.obs.groupby("section", observed=True)["log1p_total_counts"].agg(
        lambda x: float(np.expm1(np.median(x) - k * np.median(np.abs(x - np.median(x))))))
qc.to_csv(OUT / "qc_sample.csv")
print(qc.to_string(), flush=True)


# ---- pipelines
def labels(a, cluster_key=None, margin=None):
    ln = a.copy()
    ln.X = ln.layers["counts"].copy()
    sc.pp.normalize_total(ln, target_sum=1e4)
    sc.pp.log1p(ln)
    sco = pd.DataFrame(index=a.obs_names)
    for k, g in MARKERS.items():
        g = [x for x in g if x in ln.var_names]
        if g:
            sc.tl.score_genes(ln, g, score_name=k, random_state=SEED)
            sco[k] = ln.obs[k]
    z = (sco - sco.mean()) / sco.std()
    brain = a.obs["tissue"].astype(str).str.contains("brain", case=False).values
    zz = z.copy()
    zz.loc[~brain, [c for c in z.columns if c in BRAIN_ONLY]] = -np.inf
    if cluster_key:  # current: cluster mean, argmax among classes offered for the cluster's majority tissue
        per = z.groupby(a.obs[cluster_key], observed=True).mean()
        maj = a.obs.groupby(cluster_key, observed=True)["tissue"].agg(lambda t: t.value_counts().index[0]).astype(str)
        lab = {cl: per.loc[cl, [c for c in per.columns if "brain" in maj[cl].lower() or c not in BRAIN_ONLY]].idxmax() for cl in per.index}
        return a.obs[cluster_key].map(lab).astype(str)
    v = np.sort(zz.values, axis=1)  # proposed: per cell, unassigned when the best class does not stand out
    best = zz.idxmax(axis=1)
    ok = (v[:, -1] > 0) & (v[:, -1] - v[:, -2] >= margin)
    zz.add_prefix("z_").replace(-np.inf, np.nan).to_parquet(OUT / f"z_{a.uns.get('run', 'x')}.parquet")
    return best.where(ok, "unassigned")


def run(a, mode):
    a = a.copy()
    if mode == "current":  # primer.py: per-section seurat HVGs on log data, union, no scaling
        hv = set()
        for s in a.obs["section"].cat.categories:
            g = a[a.obs["section"] == s].copy()
            sc.pp.normalize_total(g, target_sum=1e4)
            sc.pp.log1p(g)
            sc.pp.highly_variable_genes(g, n_top_genes=N_HVG, flavor="seurat")
            hv |= set(g.var_names[g.var["highly_variable"]])
        genes = sorted(hv | {x for v in MARKERS.values() for x in v if x in a.var_names})
        a = a[:, genes].copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    else:
        sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat_v3", batch_key="section", layer="counts")
        a = a[:, a.var["highly_variable"]].copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        if mode == "regress":
            sc.pp.regress_out(a, ["log1p_total_counts"])
        sc.pp.scale(a, max_value=10)
    nn, res = (15, 0.5) if mode == "current" else (30, 1.0)
    if CPU:
        sc.tl.pca(a, n_comps=N_PCS, random_state=SEED)
        sc.pp.neighbors(a, n_neighbors=nn, use_rep="X_pca", random_state=SEED, key_added="raw")
        sc.tl.umap(a, neighbors_key="raw", random_state=SEED)
        a.obsm["X_umap_raw"] = a.obsm.pop("X_umap")
        import harmonypy  # scanpy 1.12.5's wrapper transposes harmonypy 2.x output wrongly (shape error), so call it directly
        z = np.asarray(harmonypy.run_harmony(a.obsm["X_pca"], a.obs, "section", random_state=SEED).Z_corr)
        a.obsm["X_pca_harmony"] = z if z.shape[0] == a.n_obs else z.T
        sc.pp.neighbors(a, n_neighbors=nn, use_rep="X_pca_harmony", random_state=SEED)
        sc.tl.umap(a, random_state=SEED)
        sc.tl.leiden(a, resolution=res, random_state=SEED, flavor="igraph", n_iterations=2)
    else:
        rsc.get.anndata_to_GPU(a)
        rsc.tl.pca(a, n_comps=N_PCS, random_state=SEED)
        rsc.pp.neighbors(a, n_neighbors=nn, use_rep="X_pca", random_state=SEED, key_added="raw")
        rsc.tl.umap(a, neighbors_key="raw", random_state=SEED)
        a.obsm["X_umap_raw"] = a.obsm.pop("X_umap")
        rsc.pp.harmony_integrate(a, key="section", basis="X_pca", adjusted_basis="X_pca_harmony", rng=SEED)
        rsc.pp.neighbors(a, n_neighbors=nn, use_rep="X_pca_harmony", random_state=SEED)
        rsc.tl.umap(a, random_state=SEED)
        rsc.tl.leiden(a, resolution=res, random_state=SEED)
        rsc.get.anndata_to_CPU(a)
    full = adata[a.obs_names].copy()  # labels score on all shared genes, as the primer does on its joint genes
    full.obs["leiden"] = a.obs["leiden"].values
    full.uns["run"] = f"{mode}_{a.n_obs}"  # distinct per run (cell counts differ)
    a.obs["cell_type"] = (labels(full, "leiden") if mode == "current" else labels(full, margin=0.1)).values
    return a


res = {f"floor{int(f * 100):03d}": run(adata[~lowq[f]], "scale") for f in (0.25, 0.3)}
PAL = dict(zip(list(MARKERS) + ["unassigned"], ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#d9d9d9"]))  # one colour per class in every run
for name, a in res.items():
    a.obs["cell_type"] = pd.Categorical(a.obs["cell_type"], categories=[c for c in PAL if c in set(a.obs["cell_type"])])
    a.uns["cell_type_colors"] = [PAL[c] for c in a.obs["cell_type"].cat.categories]
    f = float(name[5:]) / 100
    lq = lowq[f]
    pd.DataFrame({"sampled": lq.groupby(adata.obs["section"], observed=True).size(), "low_quality": lq.groupby(adata.obs["section"], observed=True).sum()}).assign(share=lambda d: (d.low_quality / d.sampled).round(3)).to_csv(OUT / f"lowq_{name}.csv")
for name, a in res.items():
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    for i, basis in enumerate(("umap_raw", "umap")):
        for j, col in enumerate(("section", "log1p_total_counts", "cell_type")):
            sc.pl.embedding(a, basis=basis, color=col, ax=axs[i, j], show=False, size=1, frameon=False,
                            title=f"{name}{' (CPU)' if CPU else ''}: {'uncorrected' if i == 0 else 'Harmony'}, {col}", sort_order=False)
    fig.savefig(OUT / f"umap_{name}.png", dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    ct = pd.crosstab(a.obs["section"], a.obs["cell_type"], normalize="index").round(3)
    ct.to_csv(OUT / f"cell_types_{name}.csv")
    a.obs.groupby("cell_type", observed=True)[["total_counts", "n_genes_by_counts"]].median().assign(n=a.obs["cell_type"].value_counts()).to_csv(OUT / f"cell_type_depth_{name}.csv")
    def entropy(s):  # section entropy of a cluster, normalised by log(number of sections): 0 one section, 1 even
        p = s.value_counts(normalize=True).values
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(adata.obs["section"].nunique()))
    lv = a.obs.groupby("leiden", observed=True).agg(n=("section", "size"), median_counts=("total_counts", "median"),
                                                    median_genes=("n_genes_by_counts", "median"), section_entropy=("section", entropy),
                                                    top_section=("section", lambda x: x.value_counts().index[0]),
                                                    label=("cell_type", lambda x: x.value_counts().index[0]),
                                                    label_share=("cell_type", lambda x: x.value_counts(normalize=True).iloc[0]))
    lv.sort_values("n", ascending=False).round(3).to_csv(OUT / f"leiden_{name}.csv")
    a.obs[["section", "total_counts", "n_genes_by_counts", "leiden", "cell_type"]].assign(
        umap_raw_1=a.obsm["X_umap_raw"][:, 0], umap_raw_2=a.obsm["X_umap_raw"][:, 1], umap_1=a.obsm["X_umap"][:, 0],
        umap_2=a.obsm["X_umap"][:, 1]).to_parquet(OUT / f"obs_{name}.parquet")
    print(name, a.shape, "clusters", a.obs["leiden"].nunique(), flush=True)
    print(ct.to_string(), flush=True)
print("done", flush=True)
