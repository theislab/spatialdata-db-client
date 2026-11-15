# Training a model on H&E crops from a catalog release

This page trains a small model on Atera slides from catalog release 2026-11-12, using only the public API. It goes
from the catalog to a cohort, writes the cohort as a loader collection, streams cell-centred H&E crops with the
spatialbatch `Loader` and trains in PyTorch. The model predicts each cell's expression of 50 genes from the 224 px
H&E crop around it. The numbers below are from one run on one GPU, so they show what the path delivers, not how good
the model is.

## The path in code

The collection is written once, on a CPU node, and validated. It holds every cell of the release and every source
feature.

```python
import sddb
from sddb.contract import validate
from sddb.loader import default_chain

cohort = sddb.Catalog(version="2026-11-12").cohort(uids=["jCLxdY1tWKmbfWOB0001"])
cohort.materialize(
    "preview-breast.zarr",
    adapters=default_chain(0.5),  # 0.5 µm per pixel, cell labels and centroids, tissue mask, stain normalisation
    workers=16,
)
assert validate("preview-breast.zarr") == []
```

The image role, the elements and each slide's pixel size come from the catalog (see {doc}`loader_atera`). Training
reads the collection with a `Loader`. Within each dataset the rows are sorted along a Morton curve of the cell
positions, so a range of rows is a compact region of the slide, and `rows=` turns that into a spatial split.

```python
import numpy as np
import torch
from spatialbatch import Loader, SpatialDatasetCollection

collection = SpatialDatasetCollection("preview-breast.zarr")
(group,) = list(collection)  # one dataset here
n = group["obsm"]["spatial"].shape[0]
n_train = int(0.8 * n) // 64 * 64  # the last 20% of rows, whole regions, is held out

loader = Loader(images=["he"], crop_px=224, batch_size=256, shuffle=True, return_index=True,
                rows=slice(0, n_train)).use_collection(collection)
data = loader.to_torch(num_workers=8)  # batch_size=None, drop_last, persistent workers, pin_memory on CUDA

for epoch in range(2):
    loader.set_rows(slice(0, n_train))  # train
    for batch in data:
        images = batch["images"]["he"]  # (256, 3, 224, 224) uint8 tensor
        X = batch["X"]                  # (256, n_features) counts, SciPy CSR
        rows = batch["index"]           # global row of each cell, to look up per-cell values
        ...
    loader.set_rows(slice(n_train, n))  # validate with the same workers
    for batch in data:
        ...
```

Things to know before writing your own loop:
- Build the loader once and reuse it across epochs. The first open of a collection checks every dataset against the
  contract; spatialbatch caches a passed check, so a later open is faster (on the six slides, 58 s for the first
  open and 14 s after, see {doc}`loader_atera`).
- `to_torch` sets up the DataLoader the way annbatch needs it. annbatch refuses several workers for a sampler that
  doesn't shuffle, so the validation rows are shuffled too (the order doesn't matter for the metrics).
- Through the DataLoader, images and indices arrive as torch tensors. `X` stays SciPy CSR, or a torch sparse tensor
  with `Loader(to="torch")`.
- `SpatialDatasetCollection(path).var` gives the gene names, and `Loader(genes=[...])` yields only those columns of
  `X`, dense float32.
- Pinned memory matters: `to_torch` turns on `pin_memory` when CUDA is available. Without it the copy to the GPU took 7.7 ms instead of 1.6 ms per batch of 256.

The full script, with the targets (log1p of counts per 10,000 Gene Expression counts, 50 most variable genes of the
training cells), the model (ResNet18 with ImageNet weights, MSE, AdamW, bf16, channels_last) and the evaluation, is
`scripts/loader_uc.py` for one slide, and `scripts/prep_v3.py` and `scripts/train_v3.py` for six (`scripts/prep_v3b.py` writes the inputs of the two hold-out tests). The scripts are as they were run, before `to_torch`, `set_rows` and
the materialize defaults existed, so they pass those settings by hand.

## One slide

The Preview FFPE breast slide has 170,057 cells. Writing its collection took 639 s with 16 workers. Two epochs on one
H100 took 2 minutes.

| | |
|---|---|
| Validation MSE (held-out last 20% of rows) | 1.039 |
| Predicting each gene's mean | 1.814 |
| Median per-gene Pearson r | 0.68 |

The best-predicted genes mark luminal epithelium (NHERF1 0.87, GATA3 0.84), so a large part of r is the model telling
epithelium from stroma.

## Six slides

The six Atera slides of the release (colorectal cancer, cervix, breast fresh frozen, breast FFPE, brain and kidney)
hold 2,310,946 cells. One whole slide, FFPE breast, is held out. The last 20% of the rows of every other slide is a
second validation set. Training uses the remaining 1,712,768 cells.

| Validation set | MSE | Predicting the training mean | Median r |
|---|---|---|---|
| New regions of the five training slides | 1.052 | 1.740 | 0.60 |
| The held-out FFPE breast slide | 1.639 | 1.445 | 0.20 |

On new regions of slides it was trained on, the model beats predicting the mean on every slide. On a slide it hasn't
seen, it does worse than predicting the training mean, even though a fresh-frozen breast slide was in training. It
still ranks some epithelial genes correctly (DSP 0.70, EPCAM 0.56), but it gets the expression levels wrong. A model
trained this way doesn't transfer to a new slide. The slides differ in preservation, sequencing depth and staining.

Two tests looked at why. Holding out the fresh-frozen breast slide instead, with the FFPE breast slide in training,
gives a median r of 0.31, but the error is still far above predicting that slide's own mean (1.795 against 1.253).
Z-scoring each gene's targets within each slide removes the slides' differences in level and scale, using the
held-out slide's own statistics. Even then the held-out slide does worse than predicting its mean (1.081 against
1.000), with a median r of 0.19. So the failure is not a simple offset between slides: what the model learns from
the H&E images of these slides does not carry over to a new slide. Each test is one run.

One of the 50 target genes, OR4F17, is an olfactory receptor with 1.5 to 2.2 counts per cell on every Atera slide in
10x's own outputs, far above every other olfactory receptor. It is likely technical but not confirmed, and it was
kept as a target because the selection rule chose it.

## Throughput

On one H100 with 224 px H&E crops at batch 256:

| torch `num_workers` | Cells per second, training |
|---|---|
| 0 (the loader alone) | 2,775 |
| 4 | 6,288 |
| 8 | 7,676 |
| 16 | 7,900 |

The loader limits training. With 8 workers it supplies about 7,500 cells per second, while the model alone runs at
about 12,000. A step of 256 cells spends about 8 ms waiting for data, 2 ms building targets on the CPU, 2 ms copying
to the GPU and 23 ms in the model. Without channels_last the model step takes 47 ms, and in fp32 33 ms.

On six slides, a training loader and two validation loaders with 8 workers each showed a Slurm `MaxRSS` of about
120 GB. That figure sums the RSS of every process and counts the pages the forked workers share once per worker, so
it overstates the memory used (see {doc}`loader_atera`). A rerun of the same setup, three loaders with 8 workers each, used about 21 GB by
PSS, the sum of each process's share of the pages. With
`set_rows`, one loader and its workers serve both splits.
