"""Six-slide loader use case, step 1 (CPU): split rows and compute the targets' statistics, reading X in row blocks.

    python -I prep_v3.py <collection> <out>

Validation A is one whole slide, dataset 3 (FFPE breast; FF breast, dataset 2, stays in training). Validation B is the
last 20% of the rows of each other dataset; rows are in Morton order per dataset, so the tail is whole spatial blocks.
Targets use the Gene Expression features present in all six datasets (absent features are zero-filled in X), each
cell's total over all of its Gene Expression features, log1p(counts / total x 1e4), and the 50 features with the
highest variance over the training rows. Writes <out>/split_v3.npz and <out>/genes_v3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import zarr
from anndata.io import read_elem

HOLD, VAL_FRAC, CHUNK, N_GENES, BLOCK = 3, 0.2, 64, 50, 50_000
col, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
r = zarr.open_group(col, mode="r")
keys = sorted(r.group_keys(), key=lambda k: int(k.split("_")[1]))
var = read_elem(r[keys[0]]["var"])
ge = (var["feature_types"].astype(str) == "Gene Expression").to_numpy()
present = np.all([read_elem(r[k]["var"])["present"].to_numpy() for k in keys], axis=0)
cand = ge & present
n = [r[k]["obsm"]["spatial"].shape[0] for k in keys]
off = np.concatenate([[0], np.cumsum(n)])
split = np.zeros(off[-1], np.int8)  # 0 train, 1 validation A (held-out slide), 2 validation B (spatial tail)
for d, k in enumerate(keys):
    if d == HOLD:
        split[off[d]:off[d + 1]] = 1
    else:
        cut = off[d] + int(round(n[d] * (1 - VAL_FRAC) / CHUNK)) * CHUNK
        split[cut:off[d + 1]] = 2
tot = np.zeros(off[-1], np.float64)
s1 = np.zeros(cand.sum()); s2 = np.zeros(cand.sum()); m = 0
for d, k in enumerate(keys):
    X = r[k]["X"]
    indptr = X["indptr"][:]
    for a in range(0, n[d], BLOCK):
        b = min(a + BLOCK, n[d])
        lo, hi = int(indptr[a]), int(indptr[b])
        blk = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], indptr[a:b + 1] - lo), shape=(b - a, len(var)))
        t = np.asarray(blk[:, ge].sum(1)).ravel()
        tot[off[d] + a:off[d] + b] = t
        tr = split[off[d] + a:off[d] + b] == 0
        if tr.any():
            v = blk[tr][:, cand].multiply(1e4 / np.maximum(t[tr], 1)[:, None]).tocsr()
            v.data = np.log1p(v.data)
            s1 += np.asarray(v.sum(0)).ravel(); s2 += np.asarray(v.multiply(v).sum(0)).ravel(); m += int(tr.sum())
    print(k, n[d], "done", flush=True)
mean = s1 / m
varg = s2 / m - mean**2
cidx = np.flatnonzero(cand)
top = cidx[np.argsort(varg)[::-1][:N_GENES]]
sym = var["original_var_name"].iloc[top].astype(str).tolist() if "original_var_name" in var else var.index[top].tolist()
np.savez(out / "split_v3.npz", split=split, total=tot, offsets=off, genes=top)
(out / "genes_v3.json").write_text(json.dumps({"genes": var.index[top].tolist(), "symbols": sym, "n_train": m,
    "n_val_slide": int((split == 1).sum()), "n_val_tail": int((split == 2).sum()), "n_candidates": int(cand.sum()),
    "offsets": off.tolist(), "datasets": keys}, indent=1))
print("train", m, "valA", int((split == 1).sum()), "valB", int((split == 2).sum()), "candidates", int(cand.sum()), "top", sym[:10])
