# spatialdata-db

Lean read client for the spatialdata-db spatial-omics collection: query the catalog in pandas and load datasets as `SpatialData` objects from anonymous S3.

> Early development (0.x): the API may change.

## Install

```bash
pip install --pre spatialdata-db
```

The client needs Python 3.12 or newer and is imported as `sddb`. `--pre` is needed while only pre-releases are published, because pip installs only stable versions without it. Extras: `spatialdata-db[viz]` for interactive Vitessce views in notebooks, `spatialdata-db[mcp]` for the `sddb-mcp` server for AI agents.

## Quickstart

```python
import sddb

cat = sddb.Catalog(version="2026-11-12")  # a dated catalog release, the same rows every time
hits = cat.query(organism="human", tissue="liver", assay="spatial-CITE-seq")
print(len(hits), "datasets")

ds = min(hits, key=lambda ds: ds.size_bytes)  # the smallest match
sdata = ds.to_sdata()  # copied to the local cache on the first call, then reused
print(sdata)

adata = hits.to_anndata()  # the tables of all matches joined into one AnnData
print(adata)
```

The documentation is at https://spatialdata-db.readthedocs.io.

## Usage

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

Catalog schema version: the client loads any catalog up to schema version `2`; a catalog cut under a newer schema is refused with a clear error asking you to `pip install -U --pre spatialdata-db`. A legacy catalog that declares no version loads unchanged.

Remote datasets: query the catalog, inspect with `elements()` (no download), or load it with `to_sdata()` (copies the store to the cache). Lazy/partial remote open is not supported yet (spatialdata upstream limitation); local stores open lazily.

## Releasing

Pushing a `v*` tag (for example `v0.2.0a1`) builds the package and publishes it to PyPI through trusted publishing (`.github/workflows/release.yml`), and the version comes from the tag. That workflow creates no GitHub release. If you also create one, use `gh release create vX.Y.Z --latest=false` so catalog releases keep resolving for older installs. The `release-guard` workflow re-marks the newest dated catalog release as Latest if one slips through.
