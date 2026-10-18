# The DB stack

The client reads a published **catalog** (one row per dataset) and then streams the
SpatialData stores it points to. This page explains what you can rely on.

## Release catalogs

Catalogs are published as dated releases named `catalog-YYYY-MM-DD`.
`sddb.Catalog()` reads the latest; pin an older one with the `version=` argument:

```python
import sddb

cat = sddb.Catalog()                      # latest release
cat = sddb.Catalog(version="2026-10-18")  # catalog-2026-10-18
```

## Validation status and tiers

Every row has a `validation_status`; queries keep only `"pass"` rows by default
(`validation=None` disables the filter). Rows also carry a `tier`: `gold`, `silver` or `bronze`,
from most to least curated. Some rows have no tier assigned yet. Not every tier is populated in every release, and gold is not guaranteed to be present. {{ tier_note }}

```python
silver = cat.query(tier="silver")
```

## Licenses

Datasets keep their original license, in the `license` column. Two boolean columns make
filtering easy: `license_noncommercial` (use is restricted to non-commercial work) and
`license_redistributable` (the data may be redistributed). Check them before reusing data
in a product or republishing it.

## Where the bytes live

Each row's `zarr_url` points to a SpatialData Zarr store on S3, readable anonymously
(no credentials). Stores are registered in the Lamin instance
[scverse/spatialdata-db](https://lamin.ai/scverse/spatialdata-db). Remote stores load eagerly:

```python
hits = cat.query(technology="Visium", tissue="lung")
smallest = min(hits, key=lambda ds: ds.size_bytes)
sdata = smallest.load(lazy=False)  # ~10 MB
```

## Versioning and drift

A dataset uid pins a specific version, but the DB does not yet retain old store bytes: when a dataset
is republished its store is replaced. `freeze()` records exactly which versions you used;
`verify()` raises `CohortDriftError` when a pinned version no longer resolves, and `check_updates()`
lists members a newer version supersedes. `download(..., pin_versions=True)` keeps your own copy.

```python
cohort = cat.query(technology="Visium")
manifest = cohort.freeze()      # exact members + deterministic hash
cohort.verify()                 # raises CohortDriftError if a pinned store no longer resolves
cohort.check_updates(cat)       # {pinned_uid: current_uid} for members a newer version supersedes
```
