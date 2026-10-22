# Changelog

## Unreleased

- `to_anndata()` reads members in parallel (`max_workers`) and can copy catalog columns into `obs` (`obs_columns`), read only some table elements (`elements`, e.g. skip duplicate count layers) and run a per-member `transform` before concatenation to bound memory.
- Documentation site (this site), with a guided tour, task guides and generated API, CLI, MCP and catalog-column reference.
- `Catalog()` reads the rolling `catalog-latest` release and `Catalog(version=...)` now resolves dated releases (#25).
