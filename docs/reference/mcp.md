# MCP tools

`sddb-mcp` serves the catalog to agents over stdio (Model Context Protocol). It needs the optional extra:

```bash
pip install "spatialdata-db[mcp] @ git+https://github.com/theislab/spatialdata-db-client@v0.1.1"
sddb-mcp
```

The `query` tool caps its output at `MAX_ROWS = 200` records; `genes` is not capped. `download` writes only to a server-controlled directory: `$SDDB_MCP_DOWNLOAD_DIR`, or a cache subdirectory when unset.

| Tool | Description | Parameters |
|---|---|---|
| `query` | Filter datasets by facets, ranges (min_obs/min_features), gene (expressing) and free-text search. | `organism`, `tissue`, `disease`, `technology`, `assay`, `tier`, `validation` (default `pass`), `expressing`, `min_fraction`, `search`, `min_obs`, `min_features` |
| `describe` | Catalog row and element shapes for one dataset uid. | `uid` |
| `genes` | Datasets expressing one or more genes (symbols or Ensembl ids; mode all=AND, any=union), optionally filtered by organism/tissue/disease/assay. Bronze datasets are included by default. | `symbols`, `organism`, `tissue`, `disease`, `assay`, `mode` (default `all`), `include_bronze` (default true) |
| `features` | Datasets measuring one or more features of any modality (genes, protein targets, metabolites; labels or ids; mode all=AND, any=union), optionally filtered by modality/feature_type and organism/tissue/disease/assay. Each row lists the typed feature hits. | `features`, `modality`, `feature_type`, `organism`, `tissue`, `disease`, `assay`, `mode` (default `all`), `include_bronze` (default true) |
| `facets` | List facet columns, or the distinct values of a facet field. | `field` (optional) |
| `cite` | BibTeX for the studies of a filtered cohort. | `organism`, `tissue`, `disease`, `technology`, `assay`, `tier`, `search`, `expressing`, `min_fraction`, `validation` |
| `download` | Plan (sizes) or fetch a filtered cohort to the server download dir. | same filters as `cite`, plus `download` (default false: plan only) |
