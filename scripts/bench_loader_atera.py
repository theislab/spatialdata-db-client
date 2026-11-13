"""Throughput of the spatialbatch Loader over a materialized Atera collection, and a figure of one batch.

Usage, on one CPU compute node, and on one GPU node for the torch sweep::

    python scripts/bench_loader_atera.py <collection> <out dir> [--epochs 2] [--max-batches N]
    python scripts/bench_loader_atera.py <collection> <out dir> --torch-workers 0,2,4,8 --max-batches 400 [--device cuda]

The loader is ``Loader(images=["he"], labels=["cells"], crop_px=224, batch_size=256, shuffle=True)`` over
``SpatialDatasetCollection(<collection>)``. Each pass is timed on its own: the first reads the collection from
lustre into the node's page cache (cold), the later ones may find it there (warm). Cells/s counts the rows yielded;
MB/s counts the bytes of the arrays yielded (the H&E and label crops, plus X's data, indices and indptr), so it
is the decoded rate, not the disk rate. Writes ``bench.json`` and ``batch.png`` into <out dir>.

With ``--torch-workers``, the same loader with ``to="torch"`` is wrapped in a ``torch.utils.data.DataLoader``
(``batch_size=None``: the Loader makes the batches; ``pin_memory`` when the device is CUDA) once per worker count,
and every batch is moved to ``--device`` (``non_blocking``, synchronised before the clock stops). Each setting reads
``--max-batches`` batches with a fresh Loader; the settings run one after another in the order given, so the later
ones may find more of the collection in the page cache. Writes ``bench_torch.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np

CROP_PX = 224
BATCH_SIZE = 256


def _nbytes(value) -> int:
    if hasattr(value, "data") and hasattr(value, "indices") and hasattr(value, "indptr"):  # scipy sparse
        return int(value.data.nbytes + value.indices.nbytes + value.indptr.nbytes)
    return int(getattr(value, "nbytes", 0))


def _batch_bytes(batch) -> int:
    total = _nbytes(batch["X"])
    for key in ("images", "labels"):
        total += sum(_nbytes(v) for v in batch[key].values())
    return total


def hardware() -> dict:
    """The node, its CPU model, the cores this job may use and the job's Slurm allocation."""
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if "model name" in line
        ),
        platform.processor(),
    )
    env = ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_PARTITION")
    return {
        "host": socket.gethostname(),
        "cpu_model": model,
        "node_cpus": os.cpu_count(),
        "usable_cpus": len(os.sched_getaffinity(0)),
        **{k.lower(): os.environ.get(k) for k in env},
    }


def bench(collection: Path, epochs: int, max_batches: int | None) -> tuple[dict, dict]:
    """Iterate the loader `epochs` times (or `max_batches` per pass) and time each pass; return the stats and a batch."""
    from spatialbatch import Loader, SpatialDatasetCollection

    t0 = time.perf_counter()
    loader = Loader(
        images=["he"], labels=["cells"], crop_px=CROP_PX, batch_size=BATCH_SIZE, shuffle=True
    ).use_collection(SpatialDatasetCollection(collection))
    stats: dict = {"setup_s": round(time.perf_counter() - t0, 2), "passes": []}
    kept = None
    for epoch in range(epochs):
        cells = nbytes = batches = 0
        t = time.perf_counter()
        first = None
        for batch in loader:
            n = batch["images"]["he"].shape[0]
            cells += n
            nbytes += _batch_bytes(batch)
            batches += 1
            if first is None:
                first = time.perf_counter() - t
            if kept is None:
                kept = batch
            if max_batches is not None and batches >= max_batches:
                break
        dt = time.perf_counter() - t
        stats["passes"].append(
            {
                "pass": epoch,
                "batches": batches,
                "cells": cells,
                "seconds": round(dt, 2),
                "first_batch_s": round(first or 0.0, 3),
                "cells_per_s": round(cells / dt, 1),
                "mb_per_s": round(nbytes / dt / 1e6, 1),
                "mb": round(nbytes / 1e6, 1),
            }
        )
        print(json.dumps(stats["passes"][-1]), flush=True)
    stats["batch_shapes"] = {
        k: {n: [list(v.shape), str(v.dtype)] for n, v in kept[k].items()} for k in ("images", "labels")
    }
    stats["batch_shapes"]["X"] = [list(kept["X"].shape), type(kept["X"]).__name__]
    return stats, kept


def _identity(batch):
    return batch


def _tensor_bytes(value) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        if value.layout == torch.sparse_csr:
            return sum(
                t.element_size() * t.numel() for t in (value.values(), value.crow_indices(), value.col_indices())
            )
        return value.element_size() * value.numel()
    return _nbytes(value)


def _to_device(batch, device):
    import torch

    def move(v):
        return v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v

    return {
        "images": {k: move(v) for k, v in batch["images"].items()},
        "labels": {k: move(v) for k, v in batch["labels"].items()},
        "X": move(batch["X"]),
    }


def bench_torch(collection: Path, workers: list[int], max_batches: int, device: str) -> dict:
    """One torch DataLoader per worker count over the spatialbatch Loader; batches moved to `device`."""
    import torch
    from spatialbatch import Loader, SpatialDatasetCollection

    stats: dict = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "settings": [],
    }
    for w in workers:
        t0 = time.perf_counter()
        loader = Loader(
            images=["he"], labels=["cells"], crop_px=CROP_PX, batch_size=BATCH_SIZE, shuffle=True, to="torch"
        ).use_collection(SpatialDatasetCollection(collection))
        setup = time.perf_counter() - t0
        dl = torch.utils.data.DataLoader(
            loader, batch_size=None, num_workers=w, collate_fn=_identity, pin_memory=device.startswith("cuda")
        )
        cells = nbytes = batches = 0
        first = None
        t = time.perf_counter()
        for batch in dl:
            moved = _to_device(batch, device)
            cells += moved["images"]["he"].shape[0]
            nbytes += sum(_tensor_bytes(v) for k in ("images", "labels") for v in batch[k].values())
            nbytes += _tensor_bytes(batch["X"])
            batches += 1
            if first is None:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                first = time.perf_counter() - t
            if batches >= max_batches:
                break
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        row = {
            "num_workers": w,
            "setup_s": round(setup, 2),
            "batches": batches,
            "cells": cells,
            "seconds": round(dt, 2),
            "first_batch_s": round(first or 0.0, 3),
            "cells_per_s": round(cells / dt, 1),
            "mb_per_s": round(nbytes / dt / 1e6, 1),
            "x_type": type(batch["X"]).__name__
            + ("/" + str(batch["X"].layout) if hasattr(batch["X"], "layout") else ""),
        }
        stats["settings"].append(row)
        print(json.dumps(row), flush=True)
        del dl, loader
    return stats


def batch_figure(batch, path: Path, n: int = 16) -> None:
    """The first `n` crops of a batch: H&E with the cell outlines, each drawn with spatialdata_plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import spatialdata as sd
    import spatialdata_plot  # noqa: F401
    from spatialdata.models import Image2DModel, Labels2DModel

    he, cells = np.asarray(batch["images"]["he"]), np.asarray(batch["labels"]["cells"])
    obs = batch.get("obs")
    cols = 8
    rows = -(-min(n, len(he)) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for i, ax in enumerate(np.ravel(axs)):
        ax.set_axis_off()
        if i >= min(n, len(he)):
            continue
        view = sd.SpatialData(
            images={"he": Image2DModel.parse(he[i], dims=("c", "y", "x"))},
            labels={"cells": Labels2DModel.parse(cells[i], dims=("y", "x"))},
        )
        (
            view.pl.render_images("he")
            .pl.render_labels("cells", outline=True, outline_color="lime", fill_alpha=0.0, contour_px=2)
            .pl.show(ax=ax, title="", colorbar=False)
        )
        ax.set_axis_off()
        if obs is not None and "in_tissue" in obs:
            ax.set_title(f"in tissue: {bool(obs['in_tissue'].iloc[i])}", fontsize=7)
    fig.suptitle(f"{min(n, len(he))} of {len(he)} crops of one batch, {CROP_PX} px at 0.5 um/px", fontsize=9)
    fig.savefig(path, dpi=72, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Benchmark, then plot one batch."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("collection", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None, help="stop each pass after this many batches")
    p.add_argument("--torch-workers", help="comma-separated DataLoader num_workers to sweep, e.g. 0,2,4,8")
    p.add_argument("--device", default="cuda", help="where the torch sweep moves each batch")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.torch_workers:
        result = {"hardware": hardware(), "collection": str(args.collection), "max_batches": args.max_batches}
        workers = [int(w) for w in args.torch_workers.split(",")]
        result.update(bench_torch(args.collection, workers, args.max_batches or 400, args.device))
        (args.out / "bench_torch.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return
    result = {"hardware": hardware(), "collection": str(args.collection), "epochs": args.epochs}
    result["max_batches"] = args.max_batches
    print(json.dumps(result), flush=True)
    stats, batch = bench(args.collection, args.epochs, args.max_batches)
    result.update(stats)
    (args.out / "bench.json").write_text(json.dumps(result, indent=1))
    batch_figure(batch, args.out / "batch.png")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
