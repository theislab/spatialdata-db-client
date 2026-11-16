"""Loader use case: predict per-cell expression from H&E crops with the public spatialdata-db + spatialbatch API.

The path a user takes: catalog release -> cohort -> loader collection (materialize) -> spatialbatch Loader -> torch.

    python -I loader_uc.py materialize <dest>                    # CPU job: write and validate the collection
    python -I loader_uc.py train <collection> <out> [--workers N] [--epochs 2] [--max-steps S]   # GPU job
    python -I loader_uc.py sweep <collection> <out>             # GPU job: data-only and train throughput per num_workers

Task: a ResNet18 (ImageNet weights) regresses log1p-normalised expression of the 50 most variable genes of the
training cells from the 224 px H&E crop centred on each cell (0.5 µm/px, so 112 µm), with MSE. Train and validation
are split by space: the collection's rows are in Morton order over 1024 px blocks, so the last 20% of rows (whole
blocks) are held out with the sampler's `mask`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

VERSION, UID = "2026-11-12", "jCLxdY1tWKmbfWOB0001"  # Preview Data: Atera, FFPE human breast cancer
ROLES = {"image": "^images/he_image$", "table": "^tables/table$", "labels": "^labels/cell_labels$"}
CROP, BATCH, N_GENES, VAL_FRAC = 224, 256, 50, 0.2


def materialize(dest: Path) -> None:
    import sddb
    from sddb.contract import validate
    from sddb.loader import default_chain

    t = {}
    t0 = time.time()
    cohort = sddb.Catalog(version=VERSION).cohort(uids=[UID])
    (record,) = list(cohort)
    t["cohort_s"] = time.time() - t0
    t0 = time.time()
    cohort.materialize(dest, adapters=default_chain(0.5), role="he", roles=ROLES, mpp_native={UID: float(record.mpp)},
                       runner="local", workers=16)
    t["materialize_s"] = time.time() - t0
    t0 = time.time()
    t["issues"] = validate(dest)
    t["validate_s"] = time.time() - t0
    print(json.dumps(t, default=str), flush=True)
    (dest.parent / f"{dest.name}.uc.json").write_text(json.dumps(t, default=str, indent=1))


def targets(collection: Path, n_train: int):
    """Top-N variable genes on the training rows, from the dataset's X and var (read once with anndata)."""
    import zarr
    from anndata.io import read_elem

    g = zarr.open_group(collection, mode="r")["dataset_0"]
    var = read_elem(g["var"])
    X = read_elem(g["X"]).tocsr()
    gene = (var["feature_types"].astype(str) == "Gene Expression").to_numpy() if "feature_types" in var else np.ones(len(var), bool)
    tot = np.asarray(X[:, gene].sum(1)).ravel()
    Xt = X[:n_train][:, gene]
    norm = Xt.multiply(1e4 / np.maximum(tot[:n_train], 1)[:, None]).tocsr()
    norm.data = np.log1p(norm.data)
    mean = np.asarray(norm.mean(0)).ravel()
    var_ = np.asarray(norm.multiply(norm).mean(0)).ravel() - mean**2
    cols = np.flatnonzero(gene)[np.argsort(var_)[::-1][:N_GENES]]
    names = var.index[cols].tolist()
    sym = var["original_var_name"].iloc[cols].astype(str).tolist() if "original_var_name" in var else names
    return cols, tot, names, sym


def make_loader(collection: Path, rows: slice, shuffle: bool, workers: int):
    from spatialbatch import Loader, SpatialDatasetCollection
    from spatialbatch.samplers import RandomSampler
    from annbatch.samplers import SequentialSampler

    # torch workers need drop_last=True (annbatch raises otherwise); the few cells of the last batch are skipped
    kw = dict(chunk_size=64, preload_nchunks=8, batch_size=BATCH, mask=rows, drop_last=workers > 0)
    sampler = RandomSampler(**kw, rng=np.random.default_rng(0)) if shuffle else SequentialSampler(**kw)
    loader = Loader(images=["he"], crop_px=CROP, batch_sampler=sampler, return_index=True).use_collection(
        SpatialDatasetCollection(collection))
    if workers == 0:
        return loader
    import torch

    return torch.utils.data.DataLoader(loader, batch_size=None, num_workers=workers, persistent_workers=True,
                                       pin_memory=True)


class GpuMon:
    """nvidia-smi samples every second into a list (utilization %, memory MiB)."""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                              "-l", "1"], stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if self._stop.is_set():
                break
            u, m = line.strip().split(", ")
            self.samples.append((time.time(), float(u), float(m)))
        p.terminate()

    def stop(self):
        self._stop.set()
        return self.samples


def run(collection: Path, out: Path, workers: int, epochs: int, max_steps: int | None, train: bool = True) -> dict:
    import resource

    import torch
    import torchvision

    import zarr

    out.mkdir(parents=True, exist_ok=True)
    n = zarr.open_group(collection, mode="r")["dataset_0"]["obsm"]["spatial"].shape[0]
    n_train = int(round(n * (1 - VAL_FRAC) / 64)) * 64  # whole chunks
    cols, tot, names, sym = targets(collection, n_train)
    dev = torch.device("cuda")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, N_GENES)
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
    colsx, totx = torch.as_tensor(cols), torch.as_tensor(tot, dtype=torch.float32)

    def to_xy(batch):
        img = torch.as_tensor(np.asarray(batch["images"]["he"])).to(dev, non_blocking=True)
        img = ((img.float() - mean) / std).contiguous(memory_format=torch.channels_last)
        X = batch["X"]
        sub = np.asarray(X[:, cols].todense()) if hasattr(X, "todense") else np.asarray(X)[:, cols]
        idx = np.asarray(batch["index"])
        y = np.log1p(sub * (1e4 / np.maximum(tot[idx], 1))[:, None]).astype(np.float32)
        return img, torch.as_tensor(y).to(dev, non_blocking=True), idx

    log = {"workers": workers, "n_cells": n, "n_train": n_train, "genes": sym, "steps": []}
    mon = GpuMon()
    t_start = time.time()
    step = 0
    t0 = time.time()
    loader = make_loader(collection, slice(0, n_train), shuffle=True, workers=workers)  # opening costs ~40 s, so once
    # a sequential sampler refuses torch workers ("Multiple workers are not supported with non-shuffled sampling")
    val_loader = make_loader(collection, slice(n_train, n), shuffle=False, workers=0) if train else None
    log["open_s"] = time.time() - t0
    for ep in range(epochs):
        model.train()
        t_wait0 = time.time()
        for batch in loader:
            t_got = time.time()
            img, y, _ = to_xy(batch)
            torch.cuda.synchronize()
            t_xy = time.time()  # targets on the CPU + host-to-device copy
            if train:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(model(img).float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                lv = float(loss)  # syncs, so compute time includes the GPU
            else:
                torch.cuda.synchronize()
                lv = None
            t_done = time.time()
            log["steps"].append({"ep": ep, "step": step, "n": int(img.shape[0]), "wait_s": t_got - t_wait0,
                                 "compute_s": t_done - t_got, "xy_s": t_xy - t_got, "loss": lv, "t": t_done - t_start})
            step += 1
            t_wait0 = time.time()
            if max_steps and step >= max_steps:
                break
        if train:
            log.setdefault("val", []).append(evaluate(model, val_loader, to_xy, step, ep))
        if max_steps and step >= max_steps:
            break
    log["gpu"] = mon.stop()
    log["wall_s"] = time.time() - t_start
    log["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    log["gpu_max_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
    s = log["steps"][5:]  # skip warm-up
    log["summary"] = {"cells_per_s": sum(x["n"] for x in s) / sum(x["wait_s"] + x["compute_s"] for x in s),
                      "wait_share": sum(x["wait_s"] for x in s) / sum(x["wait_s"] + x["compute_s"] for x in s),
                      "median_wait_s": float(np.median([x["wait_s"] for x in s])),
                      "median_compute_s": float(np.median([x["compute_s"] for x in s])),
                      "median_xy_s": float(np.median([x["xy_s"] for x in s])),
                      "gpu_util_median": float(np.median([u for _, u, _ in log["gpu"]])) if log["gpu"] else None}
    print(json.dumps({k: log[k] for k in ("workers", "open_s", "summary", "wall_s", "max_rss_gb", "gpu_max_alloc_gb")}), flush=True)
    (out / f"log_w{workers}{'' if train else '_data'}.json").write_text(json.dumps(log, default=float))
    return log


def evaluate(model, val_loader, to_xy, step, ep):
    import torch

    model.eval()
    se, cnt, preds, ys = 0.0, 0, [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in val_loader:
            img, y, _ = to_xy(batch)
            p = model(img).float()
            se += float(((p - y) ** 2).sum())
            cnt += y.numel()
            preds.append(p.cpu().numpy())
            ys.append(y.cpu().numpy())
    P, Y = np.concatenate(preds), np.concatenate(ys)
    r = [float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) if Y[:, j].std() > 0 else float("nan") for j in range(Y.shape[1])]
    base = float(((Y - Y.mean(0)) ** 2).mean())  # predicting the validation mean per gene
    model.train()
    return {"ep": ep, "step": step, "val_mse": se / cnt, "val_mse_mean_baseline": base, "pearson_r": r}



def profile(collection: Path, out: Path, steps: int = 150) -> None:
    """Time each part of a training step (CUDA-synchronised) for a few settings, plus the GPU alone on a fixed batch."""
    import torch
    import torchvision
    import zarr

    out.mkdir(parents=True, exist_ok=True)
    n = zarr.open_group(collection, mode="r")["dataset_0"]["obsm"]["spatial"].shape[0]
    n_train = int(round(n * (1 - VAL_FRAC) / 64)) * 64
    cols, tot, _, _ = targets(collection, n_train)
    dev = torch.device("cuda")
    configs = [dict(name="baseline", bs=256, cl=True, amp=True, pin=True), dict(name="no channels_last", bs=256, cl=False, amp=True, pin=True),
               dict(name="fp32", bs=256, cl=True, amp=False, pin=True), dict(name="batch 512", bs=512, cl=True, amp=True, pin=True),
               dict(name="no pin_memory", bs=256, cl=True, amp=True, pin=False)]
    res = []
    for c in configs + [dict(c, name=f"GPU only, {c['name']}", synthetic=True) for c in configs[:4]]:
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, N_GENES)
        model = model.to(dev)
        if c["cl"]:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
        if c.get("synthetic"):
            img_u8 = torch.randint(0, 255, (c["bs"], 3, CROP, CROP), dtype=torch.uint8, device=dev)
            y0 = torch.randn(c["bs"], N_GENES, device=dev)
            it = iter(lambda: None, 1)  # endless
        else:
            global BATCH
            BATCH = c["bs"]
            from spatialbatch import Loader, SpatialDatasetCollection
            from spatialbatch.samplers import RandomSampler

            s = RandomSampler(chunk_size=64, preload_nchunks=max(8, c["bs"] // 32), batch_size=c["bs"], mask=slice(0, n_train),
                              drop_last=True, rng=np.random.default_rng(0))
            L = Loader(images=["he"], crop_px=CROP, batch_sampler=s, return_index=True).use_collection(SpatialDatasetCollection(collection))
            it = iter(torch.utils.data.DataLoader(L, batch_size=None, num_workers=8, persistent_workers=True, pin_memory=c["pin"]))
        T = {"wait": [], "targets": [], "h2d": [], "fwd_bwd": []}
        for i in range(steps + 10):
            t0 = time.perf_counter()
            b = next(it)
            t1 = time.perf_counter()
            if c.get("synthetic"):
                y_cpu, t2 = None, time.perf_counter()
                img, y = img_u8, y0
            else:
                X = b["X"]
                sub = np.asarray(X[:, cols].todense())
                idx = np.asarray(b["index"])
                y_cpu = torch.as_tensor(np.log1p(sub * (1e4 / np.maximum(tot[idx], 1))[:, None]).astype(np.float32))
                t2 = time.perf_counter()
                img = torch.as_tensor(b["images"]["he"]).to(dev, non_blocking=True)
                y = y_cpu.to(dev, non_blocking=True)
            x = (img.float() - mean) / std
            if c["cl"]:
                x = x.contiguous(memory_format=torch.channels_last)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=c["amp"]):
                loss = torch.nn.functional.mse_loss(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            if i >= 10:
                for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                    T[k].append(v)
        r = {"name": c["name"], "bs": c["bs"], **{f"{k}_ms": 1e3 * float(np.mean(v)) for k, v in T.items()}}
        r["cells_per_s"] = c["bs"] * steps / sum(sum(v) for v in T.values())
        r["gpu_cells_per_s"] = c["bs"] / float(np.mean(T["fwd_bwd"]))
        print(json.dumps(r), flush=True)
        res.append(r)
        del it
    (out / "profile.json").write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["materialize", "train", "sweep", "profile"])
    p.add_argument("path", type=Path)
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None)
    a = p.parse_args()
    if a.cmd == "materialize":
        materialize(a.path)
    elif a.cmd == "profile":
        profile(a.path, a.out)
    elif a.cmd == "train":
        run(a.path, a.out, a.workers, a.epochs, a.max_steps)
    else:
        for w in (0, 4, 8, 16):
            run(a.path, a.out, w, 1, 150, train=False)  # data only: loader + host-to-device + target build
            run(a.path, a.out, w, 1, 150, train=True)

