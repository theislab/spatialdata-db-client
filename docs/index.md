# spatialdata-db

Lean read client for the spatialdata-db spatial-omics collection.

## Getting the catalog

```python
import sddb

cat = sddb.Catalog()  # central published catalog (placeholder URL, not live yet)
cat = sddb.Catalog.from_file("catalog.parquet")  # local downloaded catalog
```

`from_file` takes an absolute or relative path (or `file://` URL) and validates against the catalog schema.

```{toctree}
:maxdepth: 2
:caption: Introduction

intro/assays
intro/db-stack
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/find-and-load
guides/cohorts
guides/ml
guides/cli-and-agents
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/api
reference/cli
reference/mcp
reference/catalog-columns
```

```{toctree}
:maxdepth: 1

changelog
```
