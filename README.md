# spatialdata-db

Lean read client for the spatialdata-db spatial-omics collection: query the catalog in pandas and load datasets as `SpatialData` objects from anonymous S3.

> Early development (0.x): the API may change.

## Install

```bash
pip install spatialdata-db
```

## Usage

```python
import sddb
```

### Getting the catalog

Either way you get a `Catalog` you can `query()`:

```python
# (a) Default: the central published catalog (DEFAULT_CATALOG_URL, or $SDDB_CATALOG_URL).
#     Note: the central URL is a placeholder and not live yet.
cat = sddb.Catalog()

# (b) Local: download catalog.parquet, then point the client at it.
cat = sddb.Catalog.from_file("catalog.parquet")  # equivalent: sddb.Catalog(url="catalog.parquet")
cat.query(organism="human")
```

`from_file` accepts absolute or relative paths and `file://` URLs, and validates the file against the catalog schema.

Remote datasets: query the catalog, inspect with `elements()` (no download), or load fully with `load(lazy=False)` (copies the store to the cache). Lazy/partial remote open is not supported yet (spatialdata upstream limitation); local stores open lazily.
