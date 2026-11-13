# Training crops from whole Atera H&E slides

This page writes four whole Atera H&E slides as one loader collection on a Slurm cluster and then measures how
fast the spatialbatch `Loader` reads training crops from it. The collection follows the loader contract
({mod}`sddb.contract`): every slide is resampled to 0.5 µm per pixel, every cell has a label and a centroid, the
tissue is masked and the H&E stains are normalised to one reference fitted on all slides. The run needs a shared
filesystem, a Slurm cluster and the `loader` extra.

## Slides

The four slides are the 10x Atera v1 WTA bundles that have an H&E image and its alignment.

| uid | tissue | sample | cells | H&E scale 0 (px) | µm per H&E px |
|---|---|---|---|---|---|
| sddb_2hj2ztx47z | colorectal cancer | fresh frozen | 446,777 | 46,690 x 26,286 | 0.2739 |
| sddb_a722jfj3pq | breast cancer | fresh frozen | 485,985 | 52,963 x 33,455 | 0.2739 |
| sddb_fmjq2bpd2q | hippocampus, non-diseased | fresh frozen | 91,431 | 45,694 x 34,152 | 0.2739 |
| sddb_l5esryipaj | kidney, non-diseased | FFPE | 399,120 | 42,169 x 23,628 | 0.2739 |

The microns per H&E pixel come from each bundle. First, `experiment.spatial` gives the microns per pixel of the
`global` coordinate system (0.21249222 for all four bundles). Second, the H&E's affine onto `global` gives the
`global` pixels per H&E pixel, as `sqrt(|det A|)` of its 2 x 2 part. Their product is `mpp_native`. All four slides
are finer than 0.5 µm per pixel, so `Rasterize` resamples them down from scale 0 and nothing is upsampled.

## Commands

Each step is one Slurm job; the scripts are in `scripts/` of the client repository.

```bash
# 1. one SpatialData store per slide: the aligned H&E, the cell boundaries and the table (about 4 min each)
python scripts/atera_loader_cohort.py store <bundle>/work <uid> <out>/stores

# 2. the controller: opens the stores, writes the workflow and runs snakemake, which submits one job per rule
python scripts/atera_loader_cohort.py materialize <out>/stores <out>/atera-he.zarr --workers 2 \
    --slurm-json '{"runtime": 360, "mem_mb": 32000, "slurm_partition": "cpu_p", "qos": "cpu_normal",
                   "fit_stain": {"mem_mb": 24000, "runtime": 720}}'

# 3. checks: validate, cells in tissue, the change made by stain normalisation, a figure of the tissue masks
python scripts/atera_loader_cohort.py check <out>/stores <out>/atera-he.zarr <out>/check

# 4. throughput of the loader on one CPU node, and a figure of one batch
python scripts/bench_loader_atera.py <out>/atera-he.zarr <out>/bench --epochs 2
```

The `materialize` step calls

```python
materialize(
    members, dest,
    adapters=[Rasterize(0.5), Centroids(), TissueMask(method="weka"), StainNormalize()],
    role="he", roles={"image": "^images/he_image$", "table": "^tables/table$"}, mpp_native=mpp,
    runner="snakemake", executor="slurm", workers=2, slurm=slurm,
)
```

The chain is {func}`~sddb.loader.default_chain` with squidpy's `weka` tissue detection instead of the default
`otsu`. On the brain slide `otsu` keeps only the nuclei, because the tissue is nearly as pale as the glass, and its
area filter then drops them all, so the stain fit fails with "no tissue pixels". `weka` finds the tissue on all
four slides.

## Slurm run

The controller ran as a Slurm job of its own (4 GB), because it opens every store before it starts snakemake, and
snakemake's Slurm executor submitted one job per rule from there. Each rule job has one CPU.

| rule | jobs | wall time | peak memory |
|---|---|---|---|
| `fit_stain` (Rasterize, Centroids and TissueMask on all four slides, then the pooled stain fit) | 1 | 48 min | 6.8 GB |
| `dataset_<uid>` (the whole chain on one slide, then the write) | 4 | 15 to 25 min | 9.5 to 40 GB |
| `validate` | 1 | 2 min | 0.4 GB |

The member jobs need the most memory, and it grows with the number of nonzeros in the slide's table. The breast
table has 1.48e9 nonzeros and its job peaked at 40 GB, so a member job asks for 56 GB. The first member jobs asked
for 32 GB and were killed while they wrote the table. The run state lives beside the collection in
`atera-he.zarr.sddb/`, and a rerun skips the slides that are done.

The collection is 13 GB on disk, with 1,423,292 cells and 18,049 genes keyed by Ensembl id.

## Checks

`sddb.contract.validate` returns no issues for the collection.

Almost every cell lies on the tissue mask, which is expected, because cells are only found on tissue. The share
of the image that is tissue is the more useful check, and the figure shows the mask outlines.

| uid | cells | in tissue | tissue share of the image | mean change by stain normalisation (tissue pixels, 0 to 255) |
|---|---|---|---|---|
| sddb_2hj2ztx47z | 446,774 | 99.9% | 59% | 1.9 |
| sddb_a722jfj3pq | 485,980 | 98.7% | 50% | 1.6 |
| sddb_fmjq2bpd2q | 91,428 | 100% | 88% | 4.5 |
| sddb_l5esryipaj | 399,110 | 100% | 79% | 3.7 |

The mask follows the tissue outline on the colorectal, breast and kidney slides. On the brain slide it also covers
bands of glass around the section, so its tissue share is too high. Stain normalisation changes every slide, by 1.6
to 4.5 grey levels on average over the tissue pixels, measured against the same slide resampled without
normalisation. The reference is fitted on all four slides together. The change is small, which suggests that the
slides are stained alike, but we have not checked that.

```{image} ../_static/loader/atera-tissue-masks.png
:alt: The four H&E slides at 0.5 µm per pixel, downsampled, with the tissue mask outlined.
```

## Loader throughput

The benchmark iterates `Loader(images=["he"], labels=["cells"], crop_px=224, batch_size=256, shuffle=True)` over
`SpatialDatasetCollection(atera-he.zarr)` for two full epochs. It ran as one Slurm job with 16 cores and 32 GB on
`cpusrv96`, a node with two Intel Xeon Gold 6142M CPUs (64 hardware threads, 376 GB), reading the collection from
lustre. Each batch holds the H&E crops (256 x 3 x 224 x 224, uint8), the cell label crops (256 x 224 x 224,
uint32) and X (256 x 18,049, CSR).

| pass | cells | time | cells per second | MB per second | first batch |
|---|---|---|---|---|---|
| 1 (cold) | 1,423,292 | 1,231 s | 1,156 | 430 | 1.09 s |
| 2 (warm) | 1,423,292 | 1,271 s | 1,120 | 416 | 0.32 s |

MB per second counts the bytes of the arrays the loader returns (529 GB per epoch, mostly the label crops), so it is
the rate of decoded data, not the rate read from disk. Opening the collection took 42 s. The second pass is no
faster than the first, so the page cache does not help here, and the loader is probably limited by decoding and
cropping rather than by lustre, which we have not profiled. A shorter run on a Xeon Gold 6248R node (40 batches)
gave 1,330 cells per second.

```{image} ../_static/loader/atera-batch.png
:alt: Sixteen 224 px crops of one shuffled batch, H&E with the cell outlines.
```
