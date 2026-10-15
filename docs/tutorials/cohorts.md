---
file_format: mystnb
kernelspec:
  name: python3
  display_name: python3
---

# Build, freeze and split a cohort

`query` and `search` return a `SpatialDataCohort`: it can be listed, sliced, summarised and downloaded.
To make an experiment reproducible, freeze it, split it by study, and build a task view with an adapter:
`query -> freeze -> write/load -> split -> check -> build`.
Blocks marked `# not executed` hit S3 and are shown but not run when the docs are built.

```{code-cell} python
import sddb

cat = sddb.Catalog.from_file("../_data/catalog_sample.parquet")
cohort = cat.query(organism="human", technology="Visium")
print(cohort)
cohort.to_df()[["uid", "tissue", "disease", "size_bytes"]]
```

```{code-cell} python
total_gb = cohort.to_df()["size_bytes"].sum() / 1e9
f"{len(cohort)} stores, {total_gb:.2f} GB"
```

`summary()`, `groupby(field)`, `modalities()`, `licenses()` and `coverage()` give quick overviews,
and `citations(path)` writes the BibTeX for the studies involved. Check the size before downloading.

## Download (network, not executed here)

`download` copies every store under `dest` (resumable, parallel) and returns a `Manifest`.

```python
# not executed
manifest = cohort.download("./data")
manifest.entries[0]
# ManifestEntry(uid='...', zarr_url='s3://...', dest='data/....zarr', size_bytes=..., status='complete', sha256='...')
```

### Reproducible re-fetch

A `Manifest` records the exact URLs downloaded; `refetch` replays them into a new directory.

```python
# not executed
from sddb import Manifest

m = Manifest.read("./data/manifest.json")
m.refetch("./data_copy")
```

Only URLs are replayed, so a store republished at the same URL is not detected.

Pass `pin_versions=True` to record the catalog version; downloading into a directory pinned to a different
version then raises `ManifestVersionMismatch` unless `allow_version_change=True`.

## Freeze (offline, not executed here)

`freeze()` pins the selection to its exact members (full uids) and returns a `CohortManifest` with a
deterministic `hash`. `write` saves it as JSON (large member lists go to a parquet sidecar);
`CohortManifest.load` restores it offline, and `SpatialDataCohort.from_manifest` rebuilds the cohort
without a catalog query.

```python
# not executed
from sddb import CohortManifest, SpatialDataCohort

manifest = cohort.freeze()
manifest.write("cohort.json")
manifest = CohortManifest.load("cohort.json")  # offline reload
cohort = SpatialDataCohort.from_manifest(manifest)
manifest.hash
```

### Liveness and updates

A `zarr_url` is a "latest" pointer, not a version-pinned one. `cohort.verify()` checks that every pinned
member still resolves, and `cohort.check_updates(catalog)` reports members that a newer catalog version
supersedes. `download(strict=...)` controls whether a missing pinned store is an error.

## Split

`split` assigns whole studies (`by="study_id"`) to one fold, so no study leaks across train, val and test.
It is deterministic for a given `seed`, independent of row order, and refuses to create empty folds on
tiny cohorts. It returns `{fold: SpatialDataCohort}`; each fold carries a `SplitManifest` (pinned to the
parent cohort hash) as `.split_manifest`, which can be written and loaded.

```python
# not executed
from sddb import SplitManifest

folds = cohort.split(by="study_id", train=0.8, val=0.1, test=0.1, seed=42)
folds["train"].split_manifest.write("split.json")
split = SplitManifest.load("split.json")
```

## Task adapters

A `TaskAdapter` (protocol in `sddb.adapter`) states what a task needs and turns a store into a model input.
Concrete adapters and their execution live in method repos, not in this client. Use `cohort.sdatas` or
`cohort.load(elements=[...])` to get the data to hand to an adapter.
