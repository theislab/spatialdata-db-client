# Changelog

## Unreleased

- `freeze()` records the sha256 of the catalog file in `CohortManifest.source.sha256`. `verify()` fetches the catalog again and raises the new `CatalogChangedError` when a dated release changed after the freeze; for the rolling `catalog-latest` it only warns. If the catalog url cannot be fetched, `verify()` raises the new `CatalogUnavailableError` and does not fall back to a cached copy. `fetch_catalog` has a `fallback=` keyword for this (default True, so `Catalog()` still falls back). `from_manifest(m, catalog=...)` makes the same check against the catalog you pass. Manifests written before this change load with `sha256=None` and skip the check.
- `to_sdata`, `to_list` and `to_dict` accept `elements=` together with `dest=`. Only the named elements are copied to `dest/<uid>.zarr`, and `dest/manifest.json` records the copy with `status="partial"` and its `elements`. A later `download(dest)` fetches the whole store, and `Manifest.refetch` copies a partial entry's elements again. Manifests written before this change still load.
- Gene and feature lookups (`Catalog().genes`, `Catalog().features`, `sddb-cli genes`, `query --gene` and the MCP `genes`/`features` tools) stream the index in batches and keep only matching rows. They no longer load the whole index into pandas, so a lookup on the 128M-row gene index uses about 0.7 GB of memory instead of more than 40 GB.
- `Catalog.dataset` accepts the 16-character stem of a uid as well as the full uid; a stem resolves to the latest version.
- Downloads print their progress on stderr, and print nothing when the store is already cached.
- `sddb-cli download` ends with a summary on stderr: "downloaded N datasets (X MB) to <dest>", plus how many were already present.
- Documentation: the scverse-doc theme, a new landing page, and a "From the website to Python" page.
- `resolve(names, roles)` and `Dataset.resolve(roles)` map adapter roles (regexes) to a store's element paths, raising on no match or ambiguity.
- `table_centroids(sdata, table, image)` returns each table row's centroid in an image's scale-0 pixel space, through the spatialdata transformation graph.
- `to_anndata()` reads members in parallel (`max_workers`) and can copy catalog columns into `obs` (`obs_columns`), read only some table elements (`elements`, e.g. skip duplicate count layers) and map each member through a `TableAdapter` (`adapter`) before concatenation to bound memory; members the adapter skips are listed with their reasons in `uns["sddb"]`.
- Documentation site (this site), with a guided tour, task guides and generated API, CLI, MCP and catalog-column reference.
- `Catalog()` reads the rolling `catalog-latest` release and `Catalog(version=...)` now resolves dated releases (#25).
