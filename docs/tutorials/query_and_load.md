---
file_format: mystnb
kernelspec:
  name: python3
  display_name: python3
---

# Query the catalog and load a dataset

This tutorial runs offline against a 15-dataset sample of the real catalog
(`docs/_data/catalog_sample.parquet`). Blocks marked `# not executed` need the network
(S3) or download a full store, so they are shown but not run when the docs are built.

```{code-cell} python
import sddb

cat = sddb.Catalog.from_file("../_data/catalog_sample.parquet")
cat
```

```{note}
In real use `sddb.Catalog()` reads the central published catalog. That URL is a placeholder and
**not live yet**; until it is, download a catalog parquet and use `Catalog.from_file(path)`.
```

## Query by facet

`query` filters on facet columns (equality, or a list for "any of"). Rows that failed validation
are excluded by default.

```{code-cell} python
res = cat.query(organism="human")
res
```

```{code-cell} python
res.to_df()[["uid", "technology", "tissue", "disease", "n_obs"]]
```

```{code-cell} python
cat.query(organism="mouse", technology=["Xenium", "VisiumHD"]).to_df()[["uid", "technology", "tissue"]]
```

## Free-text search

`search` matches tokens against facet values (OR within a facet, AND across facets).

```{code-cell} python
hits = cat.search("human xenium")
hits.to_df()[["uid", "technology", "organism", "tissue"]]
```

## Pick a dataset

```{code-cell} python
d = hits[0]
print(d)
print(d.uid)
print(d.zarr_url)
print(d.viewer_url()[:100], "...")
```

`viewer_url()` is a pure string transform; no network is touched.

## Inspect and load (network, not executed here)

`elements()` reads only the store metadata from anonymous S3, so it is cheap.

```python
# not executed
d.elements()
# {'images': {...}, 'labels': {...}, 'points': {...}, 'shapes': {...}, 'tables': {...}}  (illustrative)
```

`load(lazy=False)` downloads the whole store (here ~1 GB) and returns a `SpatialData` object.
The default `lazy=True` streams chunks from S3 on access instead.

```python
# not executed
sdata = d.load(lazy=False)
sdata
```

```{note}
Gene search (`cat.genes.where_expressed(...)`) needs the published `gene_index.parquet`, which has not
been generated yet, so it is omitted for now.
```
