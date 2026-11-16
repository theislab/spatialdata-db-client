"""Six-slide use case, tests of the held-out failure, CPU step.

    python -I prep_v3b.py <collection> <prep dir>

Writes, next to split_v3.npz (same rows, totals and 50 target genes):
- split_v3_hold2.npz: the FF breast slide (dataset 2) held out whole instead of the FFPE breast slide (dataset 3, now in
  training); validation B is the Morton tail of every other slide, as before.
- zstats_v3.npz: per slide and per target gene, the mean and SD of log1p(counts / total x 1e4) over all cells of the
  slide, for per-slide z-scored targets.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import zarr

VAL_FRAC, CHUNK, BLOCK = 0.2, 64, 50_000
col, prep = Path(sys.argv[1]), Path(sys.argv[2])
z = np.load(prep / "split_v3.npz")
tot, genes, off = z["total"], z["genes"], z["offsets"]
n = np.diff(off)
split = np.zeros(off[-1], np.int8)
for d in range(len(n)):
    if d == 2:
        split[off[d]:off[d + 1]] = 1
    else:
        cut = off[d] + int(round(n[d] * (1 - VAL_FRAC) / CHUNK)) * CHUNK
        split[cut:off[d + 1]] = 2
np.savez(prep / "split_v3_hold2.npz", split=split, total=tot, offsets=off, genes=genes)
r = zarr.open_group(col, mode="r")
mu, sd = np.zeros((len(n), len(genes))), np.zeros((len(n), len(genes)))
for d in range(len(n)):
    X = r[f"dataset_{d}"]["X"]
    indptr = X["indptr"][:]
    ncol = X.attrs["shape"][1]
    s1 = np.zeros(len(genes)); s2 = np.zeros(len(genes))
    for a in range(0, n[d], BLOCK):
        b = min(a + BLOCK, n[d])
        lo, hi = int(indptr[a]), int(indptr[b])
        blk = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], indptr[a:b + 1] - lo), shape=(b - a, ncol))
        y = np.log1p(np.asarray(blk[:, genes].todense()) * (1e4 / np.maximum(tot[off[d] + a:off[d] + b], 1))[:, None])
        s1 += y.sum(0); s2 += (y**2).sum(0)
    mu[d] = s1 / n[d]; sd[d] = np.sqrt(np.maximum(s2 / n[d] - mu[d] ** 2, 1e-12))
    print(d, n[d], "mean of means", round(float(mu[d].mean()), 3), "min sd", round(float(sd[d].min()), 4), flush=True)
np.savez(prep / "zstats_v3.npz", mean=mu, sd=sd)
print("hold2 train", int((split == 0).sum()), "valA", int((split == 1).sum()), "valB", int((split == 2).sum()))
