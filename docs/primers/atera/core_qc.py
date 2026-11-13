"""Segmentation and control fields of the low-count core vs the rest of the cells, per section.

Core = Leiden 2 of the 100,000-cell "scale" run of qc_sample_check.py (its obs_scale.parquet). Reads the obs of all six Atera
tables of catalog-2026-11-12 through the client and writes core_vs_rest.csv (100k sample) and section_fields.csv (every
cell of each table: the distributions a rule on the full data would act on).

    python -I core_qc.py QCT_DIR        (CPU, from docs/primers/atera; SDDB_CACHE_DIR set)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sddb

D = Path(sys.argv[1])
FIELDS = ["cell_area", "nucleus_area", "nucleus_count", "transcript_counts", "control_probe_counts", "control_codeword_counts",
          "genomic_control_counts", "unassigned_codeword_counts", "deprecated_codeword_counts"]
o = pd.read_parquet(D / "obs_scale.parquet")
core = o.index[o["leiden"].astype(str) == "2"]

atera = sddb.Catalog(version="2026-11-12").query(technology="10x_atera")
m = atera.members
parts = []
for r in m.itertuples():
    ds = next(iter(atera.where(m["uid"] == r.uid)))
    ob = ds.to_sdata(elements=["tables/table"]).tables["table"].obs
    keep = [c for c in FIELDS + ["segmentation_method", "z_level"] if c in ob]
    x = ob[keep].copy()
    x.index = ob.index.astype(str) + "-" + r.uid
    x["section"] = o["section"].astype(str).reindex(x.index)
    x["uid"] = r.uid
    print(r.uid, r.tissue, len(x), "missing:", sorted(set(FIELDS + ["segmentation_method", "z_level"]) - set(keep)), flush=True)
    parts.append(x)
allc = pd.concat(parts)
ctrl = [c for c in FIELDS if c.endswith("_counts") and c != "transcript_counts" and c in allc]
allc["control_share"] = allc[ctrl].sum(axis=1) / (allc["transcript_counts"] + allc[ctrl].sum(axis=1)).replace(0, np.nan)
allc["unassigned_share"] = allc["unassigned_codeword_counts"] / (allc["transcript_counts"] + allc[ctrl].sum(axis=1)).replace(0, np.nan)
allc["counts_per_um2"] = allc["transcript_counts"] / allc["cell_area"].replace(0, np.nan)
allc["no_nucleus"] = allc["nucleus_count"].fillna(0) == 0
if "z_level" in allc:
    allc["z_level"] = pd.to_numeric(allc["z_level"], errors="coerce")

samp = allc.loc[allc.index.intersection(o.index)].copy()
samp["section"] = o.loc[samp.index, "section"].astype(str)
samp["group"] = np.where(samp.index.isin(core), "core", "rest")
num = ["cell_area", "nucleus_area", "transcript_counts", "counts_per_um2", "control_share", "unassigned_share", "z_level"]
t = samp.groupby(["section", "group"]).agg(n=("cell_area", "size"), **{f"median_{c}": (c, "median") for c in num if c in samp},
                                           share_no_nucleus=("no_nucleus", "mean"))
if "segmentation_method" in samp:
    seg = pd.crosstab([samp["section"], samp["group"]], samp["segmentation_method"].astype(str), normalize="index").round(3)
    t = t.join(seg.add_prefix("seg_"))
t.round(3).to_csv(D / "core_vs_rest.csv")
print(t.round(3).to_string(), flush=True)

sec = allc.groupby("uid").agg(n=("cell_area", "size"), **{f"median_{c}": (c, "median") for c in num if c in allc},
                              share_no_nucleus=("no_nucleus", "mean"))
if "segmentation_method" in allc:
    sec = sec.join(pd.crosstab(allc["uid"], allc["segmentation_method"].astype(str), normalize="index").round(3).add_prefix("seg_"))
sec.round(3).to_csv(D / "section_fields.csv")
print(sec.round(3).to_string(), flush=True)
allc[["uid", "section", "transcript_counts", "cell_area", "nucleus_count", "control_share", "unassigned_share", "counts_per_um2"]].to_parquet(D / "all_cells_fields.parquet")
print("done", flush=True)
