"""Six-slide loader use case, step 2 (GPU): the same model as loader_uc.py on the six Atera slides.

    python -I train_v3.py <collection> <prep dir> <out> [--epochs 2] [--workers 8]

Rows, totals and the 50 target genes come from prep_v3.py (split_v3.npz). Validation A is the held-out FFPE breast
slide (dataset 3), validation B the spatial tails of the other five slides. Needs spatialbatch 621320d or later
(`Loader(rows=...)`, `batch["dataset"]`).
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from spatialbatch import Loader, SpatialDatasetCollection

CROP, BATCH, NAMES = 224, 256, ["CRC FF", "cervix FFPE", "breast FF", "breast FFPE", "brain FF", "kidney FFPE"]

p = argparse.ArgumentParser()
p.add_argument("collection", type=Path)
p.add_argument("prep", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--split", default="split_v3.npz", help="split file in the prep dir (split_v3_hold2.npz holds out FF breast)")
p.add_argument("--zscore", action="store_true", help="targets z-scored per gene per slide (zstats_v3.npz)")
p.add_argument("--tag", default="v3")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
z = np.load(a.prep / a.split)
split, tot, genes = z["split"], z["total"], z["genes"]
meta = json.loads((a.prep / "genes_v3.json").read_text())
if a.zscore:
    zs = np.load(a.prep / "zstats_v3.npz")
    zmu, zsd = torch.as_tensor(zs["mean"], dtype=torch.float32), torch.as_tensor(zs["sd"], dtype=torch.float32)

collection = SpatialDatasetCollection(a.collection)


def loader(rows):
    L = Loader(images=["he"], crop_px=CROP, batch_size=BATCH, shuffle=True, drop_last=True, return_index=True, rows=rows)
    return torch.utils.data.DataLoader(L.use_collection(collection), batch_size=None, num_workers=a.workers,
                                       persistent_workers=True, pin_memory=True)


t0 = time.time()
train_dl, val_a, val_b = loader(split == 0), loader(split == 1), loader(split == 2)
open_s = time.time() - t0

dev = torch.device("cuda")
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(genes))
model = model.to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255


def to_xy(b):
    img = torch.as_tensor(b["images"]["he"]).to(dev, non_blocking=True)
    img = ((img.float() - mean) / std).contiguous(memory_format=torch.channels_last)
    idx = np.asarray(b["index"])
    sub = np.asarray(b["X"][:, genes].todense())
    y = torch.as_tensor(np.log1p(sub * (1e4 / np.maximum(tot[idx], 1))[:, None]).astype(np.float32))
    d = np.asarray(b["dataset"])
    if a.zscore:  # each slide's own mean and SD per gene, so only the within-slide pattern is left to predict
        y = (y - zmu[d]) / zsd[d]
    return img, y.to(dev, non_blocking=True), d


def evaluate(dl):
    model.eval()
    P, Y, D = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in dl:
            img, y, d = to_xy(b)
            P.append(model(img).float().cpu().numpy()); Y.append(y.cpu().numpy()); D.append(d)
    model.train()
    P, Y, D = np.concatenate(P), np.concatenate(Y), np.concatenate(D)

    def stats(m):
        p, y = P[m], Y[m]
        r = [float(np.corrcoef(p[:, j], y[:, j])[0, 1]) if y[:, j].std() > 0 else float("nan") for j in range(y.shape[1])]
        return {"n": int(m.sum()), "mse": float(((p - y) ** 2).mean()), "mse_val_mean": float(((y - y.mean(0)) ** 2).mean()),
                "mse_train_mean": float(((y - train_mean) ** 2).mean()), "median_r": float(np.nanmedian(r)), "r": r}

    out = {"all": stats(np.ones(len(D), bool))}
    for d in np.unique(D):
        out[NAMES[d]] = stats(D == d)
    return out


log = {"split": a.split, "zscore": a.zscore, "open_s": open_s, "genes": meta["symbols"], "steps": [], "val": []}
s1, cnt, train_mean = np.zeros(len(genes)), 0, None
t_start, step = time.time(), 0
for ep in range(a.epochs):
    tw = time.time()
    for b in train_dl:
        tg = time.time()
        img, y, _ = to_xy(b)
        if ep == 0:
            s1 += y.sum(0).cpu().numpy(); cnt += y.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.mse_loss(model(img).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0:  # one host sync every 25 steps for the logged loss
            log["steps"].append({"step": step, "ep": ep, "loss": float(loss), "wait_s": tg - tw, "t": time.time() - t_start})
        step += 1
        tw = time.time()
    train_mean = s1 / cnt
    te = time.time()
    v = {"ep": ep, "step": step, "train_s": te - t_start, "A_heldout_slide": evaluate(val_a), "B_spatial_tails": evaluate(val_b)}
    v["eval_s"] = time.time() - te
    log["val"].append(v)
    print(json.dumps({"ep": ep, "step": step, "A": {k: round(v["A_heldout_slide"]["all"][k], 4) for k in ("mse", "mse_val_mean", "mse_train_mean", "median_r")},
                      "B": {k: round(v["B_spatial_tails"]["all"][k], 4) for k in ("mse", "mse_val_mean", "mse_train_mean", "median_r")},
                      "eval_s": round(v["eval_s"]), "elapsed_s": round(time.time() - t_start)}), flush=True)
log["wall_s"] = time.time() - t_start
log["cells_per_s_train"] = step * BATCH / sum(v["train_s"] - (log["val"][i - 1]["train_s"] + log["val"][i - 1]["eval_s"] if i else 0) for i, v in enumerate(log["val"]))
log["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
log["gpu_max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
(a.out / f"log_{a.tag}.json").write_text(json.dumps(log, default=float))
print(json.dumps({k: log[k] for k in ("open_s", "wall_s", "cells_per_s_train", "max_rss_gb", "gpu_max_alloc_gb")}), flush=True)
