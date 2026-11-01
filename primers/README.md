# Cohort primers

## What a primer is

A primer is a curated cohort of datasets defined by a stated catalog query (facets plus an optional list of uids). A cohort needs at least 3 papers among its kept members. Each primer is described in one self-contained notebook that uses only the client and public scientific packages.

This directory is a separate Python project (`sddb-primers`). It is not part of the `sddb` client package and is not included in the client's wheel or sdist. The `primers` package holds the analysis (cohort loading, QC, embedding, mixing statistics) and the export step. The notebook `notebooks/primer.ipynb` calls these functions, so the notebook's HTML and the bundle come from one run of one codebase. Each analysis cell links the function's source at the reference that ran, and the notebook shows the pinned install line:

```bash
pip install "sddb-primers @ git+https://github.com/theislab/spatialdata-db-client@<ref>#subdirectory=primers"
```

## Running

`cohorts.yaml` defines the cohorts and `markers.yaml` the marker sets per tissue, with a citation per set. Readings name compartments only from these sets.

On the HPC, the environment is `env/pixi.toml` (rapids-singlecell on CUDA 13, checked on A100 and B300), and `hpc/*.sbatch` run the steps on compute nodes with caches on the node's local disk:

The jobs work in one workspace folder on the cluster, with `code/` (this repository, copied there by `hpc/sync.sh <ssh host> <workspace>`), `bundles/` and `logs/`. Submit them from the workspace, e.g. `cd <workspace> && sbatch code/primers/hpc/install.sbatch`, so `WORK` and the logs resolve there, or set `WORK` yourself.

1. `hpc/install.sbatch` installs the environment.
2. `hpc/panel_check.sbatch` runs `scripts/panel_check.py`, which checks every marker set against each member's gene panel before any embedding. A compartment that a panel misses needs a `fallback` in `markers.yaml`, chosen from that panel's genes.
3. `hpc/gpu_check.sbatch` prints the GPU and versions and runs PCA, Harmony and UMAP on random data, with RMM managed memory as in the runs. Pick the GPU at submit time, e.g. `sbatch --constraint=b300 code/primers/hpc/gpu_check.sbatch`.
4. `hpc/run_primer.sbatch` runs the notebook for one cohort with papermill and converts it to HTML, both into the cohort's bundle: `sbatch --export=ALL,SLUG=human-colon code/primers/hpc/run_primer.sbatch`. The source links point to the commit of the code checkout. A development run (the default) shows a banner when that commit is on no remote branch or the checkout has uncommitted changes, and `PUBLISH=1` refuses to run then. The run removes papermill's absolute paths from the notebook's metadata. `SMOKE=3` runs on the smallest member of each of 3 papers (bundle `<slug>-smoke`). `hpc/submit_all.py` prints one such command per cohort that is not `reserve`, each waiting for the previous job to end; `--submit` submits them and `--only slug1,slug2` limits the list.

`scripts/run_cohort.py` (with `hpc/run_cohort.sbatch`) builds the same bundle without the notebook.

Slurm writes the logs to `$WORK/logs`, which must exist before the first submit (`mkdir -p $WORK/logs`).

For the tests, from the repository root:

```bash
uv run --directory primers --group test pytest tests
```

The optional `gpu` extra installs rapids-singlecell. Published numbers come from the GPU run only.

## Bundle contract

Each primer writes a bundle to `<out>/<slug>/`.

- `manifest.json`:
  - slug, title, query, release;
  - member uids (kept and dropped);
  - n_datasets, n_papers, technologies;
  - n_cells_before, n_cells_after, n_flagged;
  - versions (sddb, scanpy, rapids_singlecell, spatialdata); Harmony is the rapids-singlecell GPU implementation, so its version is the rapids-singlecell version;
  - seeds, hardware, runtime_s, path, corrected and mixing stats.
- `umap.parquet`: `view` (a family name, `<family>:<technology>` when a family is split, or "joint"), `x`, `y`, `x_harmony`, `y_harmony`, `dataset_uid`, `paper`, `technology`, `tissue`, `total_counts` and `n_genes`. At most 50,000 rows per view, with an equal share per dataset.
- The manifest also records `views` (per view its `family`, `note` and `genes`: `measured`, `used`, `rule`, `size_factors`), `views_skipped` (a technology with under 50 shared genes and the reason), `gates` (with each excluded technology and its reason), `paper_inference`, `member_genes` (per kept member `n_genes` and `n_ensembl`, how many of its symbol-keyed genes also have an Ensembl id) and `embedding_h5ad_bytes`.
- Members with no cells after QC are dropped in the manifest and `cohort.parquet`, and the kept counts leave them out; fewer than 3 papers after QC stops the export.
- `qc.parquet`, `cohort.parquet`.
- `embedding.h5ad`: obs and the PCA (float16, 30 components), Harmony and UMAP embeddings, with one boolean `in_web_subsample_<view>` obs column per view (`:` read as `_`). X is empty. The export refuses a file of 2 GiB or more (GitHub's release-asset limit).
- `primer.ipynb` and `primer.html`.
