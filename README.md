# spatialdata-db

Lean read client for the spatialdata-db spatial-omics collection: query the catalog in pandas and load datasets as `SpatialData` objects from anonymous S3.

> Early development (0.x): the API may change.

## Install

```bash
pip install "spatialdata-db @ git+https://github.com/theislab/spatialdata-db-client@v0.1.1"
```

This installs the alpha release, v0.1.1. The `main` branch may be ahead of it, and you get the newest code if you leave out `@v0.1.1`.

## Usage

```python
import sddb
```

### Getting the catalog

Either way you get a `Catalog` you can `query()`:

```python
# (a) Default: the published catalog (DEFAULT_CATALOG_URL, or $SDDB_CATALOG_URL),
#     served from this repo's rolling `catalog-latest` GitHub release — live, anonymous, no account.
cat = sddb.Catalog()

# (b) Local: download catalog.parquet, then point the client at it.
cat = sddb.Catalog.from_file("catalog.parquet")  # equivalent: sddb.Catalog(url="catalog.parquet")
cat.query(organism="human")
```

`from_file` accepts absolute or relative paths and `file://` URLs, and validates the file against the catalog schema.

Remote datasets: query the catalog, inspect with `elements()` (no download), or load it with `to_sdata()` (copies the store to the cache). Lazy/partial remote open is not supported yet (spatialdata upstream limitation); local stores open lazily.

## Releasing

Cut code releases with `gh release create vX.Y.Z --latest=false` so catalog releases keep resolving for older installs. The `release-guard` workflow re-marks the newest dated catalog release as Latest if one slips through.
