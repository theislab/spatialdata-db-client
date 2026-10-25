# Changelog

## Unreleased

- `resolve(names, roles)` and `Dataset.resolve(roles)` map adapter roles (regexes) to a store's element paths, raising on no match or ambiguity.
- `table_centroids(sdata, table, image)` returns each table row's centroid in an image's scale-0 pixel space, through the spatialdata transformation graph.
- `to_anndata()` reads members in parallel (`max_workers`) and can copy catalog columns into `obs` (`obs_columns`), read only some table elements (`elements`, e.g. skip duplicate count layers) and map each member through a `TableAdapter` (`adapter`) before concatenation to bound memory; members the adapter skips are listed with their reasons in `uns["sddb"]`.
- Documentation site (this site), with a guided tour, task guides and generated API, CLI, MCP and catalog-column reference.
- `Catalog()` reads the rolling `catalog-latest` release and `Catalog(version=...)` now resolves dated releases (#25).
