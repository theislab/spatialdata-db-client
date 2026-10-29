# Changelog

## Unreleased

- `to_anndata(gene_key="ensembl")` and `gene_key="symbol"` put every member's genes under one id type before the join. Sources store genes differently (HEST keys them by Ensembl id, 10x by symbol), so the same gene used to land in two columns. Features without that id, such as control probes, are dropped, repeated ids keep the first, and one warning reports how many. The old name stays in `var["original_var_name"]`. The default `gene_key=None` leaves gene names as stored, and `to_anndata` now warns when members share fewer than half the genes of the smallest one. A member whose gene names repeat no longer makes `to_anndata` fail: the repeats get a `-1` style suffix and a warning names the member.
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
