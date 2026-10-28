# Changelog

## Unreleased

- `Catalog.dataset` accepts the 16-character stem of a uid as well as the full uid; a stem resolves to the latest version.
- Downloads print their progress on stderr, and print nothing when the store is already cached.
- `sddb-cli download` ends with a summary on stderr: "downloaded N datasets (X MB) to <dest>", plus how many were already present.
- Documentation: the scverse-doc theme, a new landing page, and a "From the website to Python" page.
- `resolve(names, roles)` and `Dataset.resolve(roles)` map adapter roles (regexes) to a store's element paths, raising on no match or ambiguity.
- `table_centroids(sdata, table, image)` returns each table row's centroid in an image's scale-0 pixel space, through the spatialdata transformation graph.
- `to_anndata()` reads members in parallel (`max_workers`) and can copy catalog columns into `obs` (`obs_columns`), read only some table elements (`elements`, e.g. skip duplicate count layers) and map each member through a `TableAdapter` (`adapter`) before concatenation to bound memory; members the adapter skips are listed with their reasons in `uns["sddb"]`.
- Documentation site (this site), with a guided tour, task guides and generated API, CLI, MCP and catalog-column reference.
- `Catalog()` reads the rolling `catalog-latest` release and `Catalog(version=...)` now resolves dated releases (#25).
