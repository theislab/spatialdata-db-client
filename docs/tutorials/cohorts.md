---
file_format: mystnb
kernelspec:
  name: python3
  display_name: python3
---

# Build and download a cohort

A `Results` object from `query` or `search` is a cohort: it can be listed, sliced, and downloaded.
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

Check the size before downloading.

## Download (network, not executed here)

`download` copies every store under `dest` (resumable, parallel) and returns a `Manifest`.

```python
# not executed
manifest = cohort.download("./data")
manifest.entries[0]
# ManifestEntry(uid='...', zarr_url='s3://...', dest='data/....zarr', size_bytes=..., status='complete', sha256='...')
```

Each entry records `uid`, `zarr_url`, `dest`, `size_bytes`, `status` (`complete` or `failed`, with `error`)
and a `sha256` of the store. The download also writes the manifest into `dest`.

## Reproducible re-fetch

A manifest replays the recorded `zarr_url`s, independent of the current catalog. Completed stores
already present are skipped.

```python
# not executed
from sddb import Manifest

m = Manifest.read("./data/manifest.json")
m.refetch("./data_copy")
```

Pass `pin_versions=True` to `download` to record the catalog version in the manifest; downloading into
a directory pinned to a different version then raises `ManifestVersionMismatch` unless
`allow_version_change=True`. Only URLs are replayed, so a store republished at the same URL is not detected.
