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

Usage examples will land with the first functional release.

Remote datasets: query the catalog, inspect with `elements()` (no download), or load fully with `load(lazy=False)` (copies the store to the cache). Lazy/partial remote open is not supported yet (spatialdata upstream limitation); local stores open lazily.
