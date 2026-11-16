# Training crops from whole Atera H&E slides

This page takes the six 10x Atera slides of catalog release 2026-11-12, writes them as one loader collection on a
Slurm cluster, and measures how fast the spatialbatch `Loader` reads training crops from it, on CPU and into a
GPU. The collection follows the loader contract ({mod}`sddb.contract`). Every slide is resampled to 0.5 µm per
pixel, every cell keeps its label id and gets a centroid, the tissue is masked, and the H&E stains are normalised
to one reference fitted on all slides. The run needs a shared filesystem, a Slurm cluster and the `loader` extra.

## Install

The `loader` extra installs squidpy, dask-image, pydantic and snakemake. spatialbatch is not on PyPI yet, so it is
installed from git, and annbatch comes from PyPI. With `uv pip install`, pass `--no-sources`: spatialbatch's own
`pyproject.toml` points annbatch at a local checkout (`../annbatch`), which uv would otherwise try to follow.

```bash
pip install "spatialdata-db[loader]" "annbatch>=0.3.1" \
    "spatialbatch @ git+https://github.com/timtreis/spatialbatch@feat/loader-api"
# or: uv pip install --no-sources "spatialdata-db[loader]" "annbatch>=0.3.1" "spatialbatch @ git+..."
python -c "import sddb.loader, spatialbatch, annbatch; print(annbatch.__version__)"  # 0.3.1 here
```

The runs on this page used annbatch 0.3.1, zarr 3.4.2, torch 2.15.0 and Python 3.12.

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
on the shared filesystem, because `materialize` copies each member's store from the release into it once, and the
Slurm jobs open the copy from there. It copies only the elements the roles select (here the H&E, the table and the
cell labels) and the elements the table annotates, under `$SDDB_CACHE_DIR/elements/<uid>/`. The run on this page was
made before that change and copied the whole stores (306 GB) into `$SDDB_CACHE_DIR/stores/<uid>.zarr`.

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

The `--slurm-json` keys are snakemake resources, which the snakemake Slurm executor plugin turns into `sbatch` flags
(plugin 2.8.0, `submit_string.py`):

| key | `sbatch` flag |
|---|---|
| `slurm_partition` | `-p` |
| `slurm_account` | `-A` |
| `qos` | `--qos` |
| `runtime` (minutes) | `-t` |
| `mem_mb`, `mem_mb_per_cpu` | `--mem`, `--mem-per-cpu` |
| `slurm_extra` | added as given, e.g. `"'--qos=cpu_normal'"` |

`fit_stain` holds the resources of that one job. The partition and the QOS names are this cluster's, so replace
them with yours. Where the cluster requires a QOS and none is given, every job fails at submission with
`sbatch: error: Batch job submission failed: Invalid qos specification`. Without `slurm_account` the plugin logs
"No SLURM account given, trying to guess" and asks `sacct`; pass the account where the cluster needs one. Run the
controller itself as a job (or in a terminal that stays open), because it waits for all the jobs it submits: on this
page that took 4 h 15 min.

The `materialize` step calls

```python
cohort.materialize(dest, adapters=default_chain(0.5), runner="snakemake", executor="slurm", workers=1, slurm=slurm)
```

`role`, `roles` and `mpp_native` default from the catalog:

- `role="he"`, because every member has an H&E image (`hne_image`);
- the element roles of 10x Atera: `^images/he_image$`, `^tables/table$` and `^labels/cell_labels$`;
- each record's `mpp`. It is checked against the pixel size the store records: spatialdata-io's Atera reader stores
  `pixel_size` (µm per `global` unit), which times the H&E's scale onto `global` gives its µm per pixel. A relative
  difference above 1e-6 raises. On the four Atera v1 stores the two agree to 4e-16. The two Preview stores record no
  pixel size, so their catalog value is taken unchecked.

Each dataset's contract says which case applies (`mpp_source`: `"catalog_checked_store"`, `"catalog_unchecked"`,
`"store_pitch_fit"` for Visium, see "Visium slides" below, or `"given"` for a value passed as
`mpp_native={uid: ...}`). The keyword arguments override the defaults: the run above passed them explicitly, with the
same values.

With `role="he"` every member's image must be 3 channels of uint8. A few IF slides carry `hne_image` in the catalog,
and for those `materialize` stops before it writes anything, with
`ValueError: <uid>: role 'he' needs a 3-channel (RGB) uint8 image, but '<image>' has 4 channels of uint16`, or
`3 channels of uint16`. Pass `role="if"` for them, or leave them out of an H&E cohort. Some Visium stores of release
2026-11-12 hold their H&E as RGBA (4 channels of uint8). `materialize` reads the alpha of a downsampled level: when
it is constant, the alpha is dropped and its value recorded in the contract (`he_alpha_dropped`); otherwise it
raises. An IF image saved as RGBA with a constant alpha passes this check, so look at the images of a cohort before
you train on it.

Each dataset also gets `obs["total_counts"]` (the sum over the Gene Expression features), and `var["mean_counts"]`
and `var["var_counts"]` (the mean and the variance with `n - 1` degrees of freedom over its cells), flagged by
`totals` in its contract. The collection on this page was written before that, so it has no `totals`, and its
`obs["total_counts"]` is the one from the source table.

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

At about 1,300 cells per second one epoch over the six slides takes about 30 minutes, so run it as a job, not in an
interactive session.

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

### H&E only and with labels

Most of the bytes of a batch above are the label crops (uint32, four times the H&E's uint8). The same loader
without `labels` reads more cells per second:

| crops | setup | DataLoader workers | cells per second |
|---|---|---|---|
| H&E + labels | this page, CPU job, 16 cores, six slides | 0 | 1,319 |
| H&E + labels | this page, H100 job, 16 cores, six slides, 400 batches | 8 | 4,639 |
| H&E | 16 cores of a GPU node (the GPU unused), six slides, 400 batches counted from the start of the workers | 8 | 3,749 |
| H&E | as above, `return_annotations=[]` | 8 | 4,163 |
| H&E | training use case, one H100, one slide (FFPE breast Preview), 150 steps, data only | 0 | 3,567 |
| H&E | as above | 8 | 7,680 |
| H&E | as above, ResNet18 training loop, bf16, `channels_last` | 8 | 7,676 |
| H&E | training use case, one H100, six slides, 2 epochs of training | 8 | 6,393 |

The last four rows were measured in the training use case ({doc}`loader_training`). It ran on one H100 80 GB with
crops of 224 px and batches of 256, and it also gives the setup and the time per step. The rows for six slides
without labels are ours. They count the start of the eight workers and read the
collection cold from lustre. The rows differ in node, slides and count, so compare them within a setup, not across.

### Reading the collection for training

{doc}`loader_training` trains a model on this collection. What it relies on:

- Open the loader once and iterate it many times. `use_collection` checks every dataset against the contract
  ({func}`~sddb.contract.validate_dataset`) and reads every dataset's `obs`. On this collection the first open of
  the six slides took 58 s, of which 39 s was the contract check. spatialbatch then records the pass under a hash
  of the collection's consolidated metadata, the datasets' contracts (uid and version included) and the sddb
  version, in the process and in `$SPATIALBATCH_CACHE_DIR` (default `~/.cache/spatialbatch`). A later open, in the
  same process or a new one, took 14 s, and 7.6 to 9.7 s with `return_annotations=[]`, which skips
  `obs`. Before the cache a reopen took 62 s.
- `loader.to_torch(num_workers=8)` returns the torch `DataLoader`. It sets `batch_size=None` (the loader makes the
  batches), `pin_memory` when CUDA is available, and persistent workers. With several workers annbatch needs the
  sampler to drop its last incomplete batch, so `to_torch` sets `drop_last` on a sampler the loader built, and each
  worker then skips at most `batch_size - 1` rows. annbatch refuses several workers for a sampler that does not
  shuffle.
- Through a `DataLoader` the arrays of a batch (the crops, `index`, `dataset`, `coords`) arrive as torch tensors with
  any number of workers. `X` stays a SciPy CSR matrix unless the loader is built with `to="torch"`, which yields a
  torch sparse tensor.
- `Loader(genes=[...])` yields only those columns of `X`, dense float32, in the order given, with the matching rows
  of `batch["var"]`. A name is a var name (here the Ensembl id) or an `original_var_name` (the gene symbol).
  `SpatialDatasetCollection(path).var` is the collection's var, with `gene_ids`, `feature_types` and
  `original_var_name`.
- `loader.set_rows(rows)` switches the rows between epochs, so one loader and its workers serve the training and the
  validation rows. Persistent workers read the new rows from a shared mask when the next epoch starts. For
  validation, pass `evaluate=True`. The pass then yields every row exactly once with any number of workers: each
  worker reads a contiguous share of the rows in order, and nothing is shuffled or dropped, so some batches are
  shorter than `batch_size`. Without it, a pass with 8 workers and batches of 256 skips up to 8 × 255 = 2,040 rows,
  a different set each time.

### Memory

Each worker of a `DataLoader` is a fork of the main process. It shares the main process's pages until it writes to
them, and Slurm's `MaxRSS` counts those shared pages once per process. On this cluster the accounting
(`jobacct_gather/linux`, without `UsePss`) sums the RSS of the job's processes, while the memory limit is the
job's cgroup, which counts a shared page once. One loader over the six slides with eight persistent workers,
H&E only, 400 batches:

| | per worker | all workers |
|---|---|---|
| RSS | 2.92 GB | 23.4 GB |
| PSS (shared pages split between the processes) | 1.02 GB | 8.2 GB |
| private, written | 0.77 GB | 6.2 GB |

The main process had an RSS of 2.9 GB, and the job's `MaxRSS` was 26.0 GB: the sum of the RSS. The memory the
job used was about 9 GB, the sum of the PSS. The 121.6 GB `MaxRSS` of the training use case's three loaders with eight
workers each is most likely the same sum, which counts the shared pages once per worker; that job was not measured
this way. The largest part of the loader's own state is `obs` (507 MB for
the six slides). With `return_annotations=[]` the main process held 0.87 GB, a worker 1.22 GB RSS and 0.80 GB PSS,
and the sum of the RSS fell to 10.7 GB. Most of a worker's private memory is its batch buffers
and the libraries it has loaded. Python objects took 0.17 GB (`tracemalloc`).

## Visium slides

The same path works for Visium v1/v2 slides, with three differences.

- There are no default roles for Visium, so pass them. spatialdata-io's visium reader names the hires image
  `<id>_hires_image` and the full-resolution image, when the bundle has one, `<id>_full_image`. Use one of them:
  `roles={"image": "^images/.*_hires_image$", "table": "^tables/table$"}`, or `.*_full_image$`. There are no labels.
  The table annotates the spot circles, and `Rasterize` draws each circle into `labels/cells`.
- The µm per pixel is fitted on the store's own spots: a least-squares fit of the spot centres on their array
  positions, with 100 µm between neighbouring spots of a row (one `array_col` step is 50 µm), as spatialdata-io's
  visium reader does. `mpp_source` is `"store_pitch_fit"`. The fit raises when the spots are not on a v1/v2 array or
  fit it to worse than 5 µm (median). The catalog `mpp` of Visium records up to release 2026-11-12 is
  55 µm / `spot_diameter_fullres`, and `spot_diameter_fullres` is a display size of the spot (about 65 to 70 µm), not
  the 55 µm spot. So the catalog value is 1.18 to 1.27 times too small, and `materialize` never uses it for those
  releases. In later releases a catalog value more than 2% off the fit raises. A Visium store without
  `array_col`/`array_row` in its table gets no mpp (native grid) unless you pass `mpp_native`.
- The fit is for technology `10x_visium` only. The catalog `mpp` of a Visium HD record (Space Ranger's
  `microns_per_pixel`, within 0.1% of the array fit on the slides checked) is taken as is (`"catalog_unchecked"`).

```python
cohort = sddb.Catalog(version="2026-11-12").query(technology="10x_visium")
cohort.materialize(
    "visium-he.zarr",
    adapters=default_chain(1.0),
    roles={"image": "^images/.*_hires_image$", "table": "^tables/table$"},
)
```

## Local or derived stores

`materialize(..., stores={uid: path})` opens a local store in place of a member's catalog store, for example a copy
on a faster disk or a store derived from it (Visium HD bins summed into spots). The member stays the catalog record,
so its catalog columns, the defaults and the cohort's release still go into the contract (`catalog_release`). Each
uid must be a member of the cohort.

```{image} ../_static/loader/atera-batch.png
:alt: Sixteen 224 px crops of one shuffled batch, H&E with the cell outlines.
```
