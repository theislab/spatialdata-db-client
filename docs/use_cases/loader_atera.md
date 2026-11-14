# Training crops from whole Atera H&E slides

This page takes the six 10x Atera slides of catalog release 2026-11-12, writes them as one loader collection on a
Slurm cluster, and measures how fast the spatialbatch `Loader` reads training crops from it, on CPU and into a
GPU. The collection follows the loader contract ({mod}`sddb.contract`). Every slide is resampled to 0.5 µm per
pixel, every cell keeps its label id and gets a centroid, the tissue is masked, and the H&E stains are normalised
to one reference fitted on all slides. The run needs a shared filesystem, a Slurm cluster and the `loader` extra.

## Slides

The cohort is every record of the release with technology `10x_atera`, and each one has an H&E image
(`images/he_image`), cell labels (`labels/cell_labels`) and a table (`tables/table`).

```python
import sddb

catalog = sddb.Catalog(version="2026-11-12")
cohort = catalog.query(technology="10x_atera")  # 6 records
mpp = dict(zip(cohort.to_df()["uid"], cohort.to_df()["mpp"]))
```

| dataset | slide | cells | features | µm per pixel | store |
|---|---|---|---|---|---|
| 0 | Fresh Frozen Human Colorectal Cancer | 446,777 | 24,677 | 0.2739 | 74.0 GB |
| 1 | Preview Data: FFPE Human Cervical Cancer | 717,576 | 18,028 | 0.2741 | 43.6 GB |
| 2 | Fresh Frozen Human Breast Cancer | 485,985 | 24,677 | 0.2739 | 87.8 GB |
| 3 | Preview Data: FFPE Human Breast Cancer | 170,057 | 18,028 | 0.2741 | 27.0 GB |
| 4 | Fresh Frozen Human Brain | 91,431 | 24,677 | 0.2739 | 39.7 GB |
| 5 | FFPE Human Kidney | 399,120 | 24,674 | 0.2739 | 34.0 GB |

In total 2,310,946 cells and 306 GB of stores. The dataset numbers are the order of the collection.

The microns per H&E pixel are the catalog's `mpp` column. For the four Atera v1 slides they equal the values that
the first version of this page derived from each bundle (the `experiment.spatial` pixel size times the scale of the
H&E's affine onto `global`). All six slides are finer than 0.5 µm per pixel, so `Rasterize` resamples them down and
nothing is upsampled.

The four Atera v1 tables annotate `cell_labels`, so each row's label id is its instance id. The two WTA Preview
slides were read with spatialdata-io's xenium reader, whose table annotates `cell_circles` and keeps each row's
value in `cell_labels` in `obs["cell_labels"]`. {meth}`~sddb.loader.LoaderItem.from_sdata` reads that column and
checks it against the labels on a sample of rows. The Preview tables have 18,028 Gene Expression features and float32
counts. The Atera v1 tables have 24,674 or 24,677 features, Gene Expression plus control probes and codewords, with
int32 counts.

## Commands

Both steps run as Slurm jobs, and the scripts are in `scripts/` of the client repository. `SDDB_CACHE_DIR` must be
on the shared filesystem: `materialize` copies each member's whole store from the release into
`$SDDB_CACHE_DIR/stores/<uid>.zarr` once, and the Slurm jobs open it from there.

```bash
export SDDB_CACHE_DIR=<shared dir>/cache

# 1. the controller: copies the stores, writes the workflow and runs snakemake, which submits one job per rule
python scripts/atera_loader_cohort.py materialize <out>/atera-he.zarr --version 2026-11-12 --workers 1 \
    --slurm-json '{"runtime": 360, "mem_mb": 80000, "slurm_partition": "cpu_p", "qos": "cpu_normal",
                   "fit_stain": {"mem_mb": 24000, "runtime": 720}}'

# 2. checks: validate, compare each dataset with its catalog record, draw the tissue masks
python scripts/atera_loader_cohort.py check <out>/atera-he.zarr <out>/check

# 3. throughput on one CPU node, and a torch DataLoader sweep on one GPU node
python scripts/bench_loader_atera.py <out>/atera-he.zarr <out>/bench --epochs 2
python scripts/bench_loader_atera.py <out>/atera-he.zarr <out>/bench_gpu --torch-workers 0,2,4,8 --max-batches 400
```

The `materialize` step calls

```python
cohort.materialize(
    dest,
    adapters=default_chain(0.5),
    role="he",
    roles={"image": "^images/he_image$", "table": "^tables/table$", "labels": "^labels/cell_labels$"},
    mpp_native=mpp,
    runner="snakemake", executor="slurm", workers=1, slurm=slurm,
)
```

The controller ran for 4 h 15 min in a 16 GB job. Snakemake submitted one `fit_stain` job (24 GB requested) and
then the six members one at a time (`workers=1`, 80 GB requested each), and a final `validate` job:

| job | wall time | peak memory |
|---|---|---|
| `fit_stain` (all six slides) | 31 min 42 s | 4.3 GB |
| dataset 3, FFPE breast Preview | 11 min 7 s | 5.8 GB |
| dataset 5, kidney | 19 min 32 s | 8.5 GB |
| dataset 0, colorectal | 26 min 36 s | 16.0 GB |
| dataset 2, FF breast | 34 min 38 s | 17.6 GB |
| dataset 1, cervical Preview | 14 min 40 s | 10.2 GB |
| dataset 4, brain | 26 min 43 s | 4.7 GB |
| `validate` | 2 min 54 s | 2.2 GB |

The colorectal job waited 71 minutes in the Slurm queue before it started, which accounts for most of the gap between
the sum of the jobs (2 h 48 min) and the controller's time. No member needed more than 18 GB, so 24 GB per member
would do. Most of `fit_stain`'s time is the resampling and the centroids of all six slides; the tissue detection
and the stain fit themselves take seconds (see below). The collection is 17.7 GB.

## Cells outside the H&E image

On the four Atera v1 slides the morphology image, and with it the cell labels, reaches past the H&E, so some cells
lie outside the H&E image. The written grid is the bounding box of the H&E's footprint, and every H&E is rotated a
little against the morphology image (0.1° to 0.4° for the Atera v1 slides and about 0.56° for the Preview slides, from
the H&E's affine in each store), so the grid's corners are fill. Every cell keeps its true centroid in
`obsm["spatial"]`, also when that lies outside the grid. `obs["outside_image"]` is True for each cell whose centroid
is not on the H&E's footprint, which is the quadrilateral of the H&E's scale-0 extent mapped onto the grid with the
affine that `Rasterize` resamples the image with. A cell over a fill corner of the grid is flagged too, and so is a
cell beyond the grid. The contract counts the flagged cells as `n_outside_image`, `validate` accepts a centroid
outside the grid only for a flagged cell, and these cells have `in_tissue` False. The spatialbatch loader pads the
part of a crop that leaves the image, so a crop of such a cell is partly or fully fill.

| dataset | slide | cells outside the H&E | of them beyond the grid |
|---|---|---|---|
| 0 | colorectal | 3 | 3 |
| 2 | FF breast | 3 | 3 |
| 4 | brain | 1 | 1 |
| 5 | kidney | 15 | 10 |

The two Preview slides have none. A cell beyond the grid has no pixel in `labels/cells`, so its centroid comes from
the source; `n_centroid_fallback` equals the count beyond the grid on every slide. The other five kidney cells lie
over a fill corner of the grid and keep their pixel centroid.

## Tissue mask

{class}`~sddb.loader.TissueMask` runs squidpy's tissue detection
({func}`~squidpy.experimental.im.detect_tissue`) on the coarsest level of each slide's H&E pyramid, the level
squidpy's `scale="auto"` reads. For all six slides that is level 4, between 2287 × 2242 and 3310 × 2090 pixels.
The mask squidpy finds there is resampled onto the 0.5 µm grid with nearest neighbour and the same affine as the
image, and is 0 outside the H&E. The step reads only that level, so on the prototype slide (the FFPE breast
Preview) it took 39 s, of which 14 s are the detection. Its default, which {func}`~sddb.loader.default_chain` uses,
is squidpy's `weka` method, a random forest trained on the corners of the level as glass and on the pixels least
like them as tissue. Three settings differ from squidpy's defaults:

- `method_params={"rng": 0}`, so that a rerun gives the same mask. squidpy's default seeds the forest from the
  operating system. The detection also runs with one BLAS/OpenMP thread: with more threads the same level gave a
  mask that differed in a few pixels.
- `corner_size_pct=0.05`, corner boxes of 5% of the height and width. With squidpy's 1%, the mask of the brain
  slide had a fringe of glass along its right and top edges.
- `corners_are_background="auto"`. A corner counts as glass when the mean grey value of its box is closer to that
  of the brightest corner box than to the median grey value of the level. The tissue of the cervical Preview slide
  covers its top-right corner, and the rule leaves that corner out; on the other five slides all four corners are
  glass.

squidpy's default `otsu` gives an empty mask on the brain slide: its tissue is almost as pale as the glass, so
`otsu` keeps only the nuclei, and the area filter then drops them. The first version of this page used `weka` with
squidpy's other defaults, and on the brain slide that mask covered bands of glass (88% of the image).

{class}`~sddb.loader.StainNormalize` fits one Macenko reference with
{func}`~squidpy.experimental.im.fit_stain_reference` on the same level of all six slides, with the masks found
there; the fit took 0.8 s on one slide. It then normalises each written image with
{meth}`~squidpy.experimental.im.StainFit.transform`, which takes the slide's own stain statistics from that level and
leaves the pixels outside the mask unchanged.

```{image} ../_static/loader/atera-tissue-masks.png
:alt: The six H&E slides with the outline of their tissue mask and the share of the image that is tissue.
```

## Checks

`check` runs {func}`~sddb.contract.validate` on the collection (no issues) and compares each dataset with its
catalog record:

| dataset | cells = catalog | features measured / filled | centroid from the source | in tissue (cells) | tissue (image) | stain change |
|---|---|---|---|---|---|---|
| 0 colorectal | yes | 24,677 / 2 | 3 | 99.32% | 57.4% | 2.36 |
| 1 cervical Preview | yes | 18,028 / 6,651 | 0 | 100.00% | 80.0% | 2.81 |
| 2 FF breast | yes | 24,677 / 2 | 3 | 99.50% | 50.4% | 1.26 |
| 3 FFPE breast Preview | yes | 18,028 / 6,651 | 0 | 99.97% | 71.9% | 3.11 |
| 4 brain | yes | 24,677 / 2 | 1 | 99.97% | 70.2% | 5.14 |
| 5 kidney | yes | 24,674 / 5 | 10 | 99.91% | 77.1% | 3.76 |

The var index is the union of the six slides, 24,679 features. A feature a slide did not measure is 0 in its `X` and
False in `var["present"]`: the Preview slides lack 6,651 features of the Atera v1 panel (its control probes and
codewords among them), and the Atera v1 slides lack two or five features of the union. Nearly every cell lies on its mask,
so `in_tissue` is close to 1; the share of the image that is tissue is the more useful check, and the figure shows
the mask outlines. The stain change is the mean absolute change of a tissue pixel (0-255) made by the
normalisation.

## Loader throughput

The CPU benchmark iterates `Loader(images=["he"], labels=["cells"], crop_px=224, batch_size=256, shuffle=True)`
over `SpatialDatasetCollection(atera-he.zarr)` for two full epochs. It ran as one Slurm job with 16 cores and 32 GB
on a node with two Intel Xeon Gold 6136 CPUs, reading the collection from lustre. Each batch holds the H&E crops
(256 x 3 x 224 x 224, uint8), the cell label crops (256 x 224 x 224, uint32) and X (256 x 24,679, CSR).

| pass | cells | time | cells per second | MB per second | first batch |
|---|---|---|---|---|---|
| 1 (cold) | 2,310,946 | 1,752 s | 1,319 | 494 | 0.68 s |
| 2 (warm) | 2,310,946 | 1,757 s | 1,315 | 493 | 0.48 s |

MB per second counts the bytes of the arrays the loader returns (866 GB per epoch, mostly the label crops), so it is
the rate of decoded data, not the rate read from disk. Opening the collection took 67 s. The second pass is no
faster than the first, so the page cache does not help here; we have not profiled whether decoding, cropping or
lustre limits it.

The GPU benchmark wraps the same loader (with `to="torch"` and `drop_last=True`, which annbatch requires for more
than one worker) in a torch `DataLoader(batch_size=None)` and moves every batch to the GPU, 400 batches (102,400
cells) per setting. It ran as one Slurm job with 16 cores of an Intel Xeon Platinum 8480+ node and one NVIDIA H100
80GB (torch 2.15.0, CUDA 13.2). X arrives as a sparse CSR tensor.

| DataLoader workers | time | cells per second | MB per second | first batch |
|---|---|---|---|---|
| 0 | 54.1 s | 1,893 | 710 | 0.74 s |
| 2 | 39.4 s | 2,602 | 976 | 0.59 s |
| 4 | 26.4 s | 3,876 | 1,453 | 1.11 s |
| 8 | 22.1 s | 4,639 | 1,738 | 0.95 s |

Eight workers read 2.4 times as fast as the loader in the main process. The H100 job's node differs from the CPU
job's, so the rows with 0 workers of the two tables are not a CPU/GPU comparison.

```{image} ../_static/loader/atera-batch.png
:alt: Sixteen 224 px crops of one shuffled batch, H&E with the cell outlines.
```
