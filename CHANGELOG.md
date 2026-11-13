# Changelog

## 0.2.0a1

First release on PyPI (`pip install --pre spatialdata-db`, import `sddb`). Releases up to v0.1.4 were installed from GitHub only.

- `Catalog()` reads the published catalog, `Catalog(version="YYYY-MM-DD")` a dated release and `Catalog.from_file()` a local `catalog.parquet`.
- `Catalog.query()` filters by facets such as organism, tissue, technology and assay, by license, by free text and by expressed genes. `Catalog.genes` and `Catalog.features` index the measured features.
- `Catalog.dataset()` looks a dataset up by full uid, 16-character stem or `canonical_key`. `Dataset.elements()` lists its elements without a download, and `Dataset.to_sdata()` copies the store to the local cache and opens it as `SpatialData`, optionally only some elements or into a folder of your choice.
- `Catalog.collection()` and `Catalog.cohort()` build a `SpatialDataCohort`. A cohort downloads its members, loads them as `SpatialData`, joins their tables into one `AnnData` with `to_anndata()`, counts licenses and writes the BibTeX citations of its studies.
- `SpatialDataCohort.freeze()` records a cohort as a `CohortManifest`, and `split()` makes a deterministic, study-safe train/val/test split recorded as a `SplitManifest`.
- `sddb-cli` runs queries, facet listings, downloads, gene lookups and citations from the shell.
- `sddb-mcp` serves the catalog to AI agents over the Model Context Protocol (extra `mcp`).
- The `viz` extra adds interactive Vitessce views in notebooks.
