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

## Freeze (network, not executed here)

`freeze()` pins the selection to its exact members, reads each store's fingerprint, and returns a
`FrozenCohort` with a deterministic `hash`. `write` saves it as JSON (large member lists go to a parquet
sidecar); `FrozenCohort.load` restores it offline, without a catalog.

```python
# not executed
from sddb import CohortDriftError, FrozenCohort

frozen = cohort.freeze()
frozen.write("cohort.json")
frozen = FrozenCohort.load("cohort.json")  # offline reload
frozen.hash, frozen.members
```

### Verify-on-read

A `zarr_url` is a "latest" pointer, not a version-pinned one. So reading a frozen cohort's data
(`frozen.to_anndata()`, `frozen.build(...)`) re-checks each object's fingerprint first and raises
`CohortDriftError` if a store changed since the freeze. Pass `verify=False` to skip the check.
A live `SpatialDataCohort.to_anndata()` is unfrozen and performs no such check.

```python
# not executed
adata = frozen.to_anndata()  # table=None: default table of each member
```

## Split

`split` assigns whole studies (`by="study_id"`) to one fold, so no study leaks across train, val and test.
It is deterministic for a given `seed`, independent of row order, and refuses to create empty folds on
tiny cohorts. The `SplitManifest` records the parent cohort hash, and can be written and loaded.

```python
# not executed
from sddb import SplitManifest

split = frozen.split(by="study_id", train=0.8, val=0.1, test=0.1, seed=42)
split.write("split.json")
split = SplitManifest.load("split.json")
```

## Check and build with a task adapter

A `TaskAdapter` (protocol in `sddb.adapter`) states what a task needs (`requirements`), validates metadata
(`validate_meta`) and data (`validate`), and turns a store into a model input (`build`). Concrete adapters
live in method repos, not in this client; `my_adapter` below stands for one of them.

`check` reports compatibility from metadata (add `deep=True` to also open the stores). `build` returns a lazy
task view; with `materialize=True` each member is built once and provenance is written to `out`. A `split`
must derive from the same cohort, otherwise `ValueError` is raised.

```python
# not executed
report = frozen.check(my_adapter)
print(report)

view = frozen.build(my_adapter, split=split, materialize=True, out="./task")
for uid, load in view:
    sdata = load()
```
