# WP-A — Richer catalog querying + first-class gene API (design)

> Self-contained: a fresh agent can execute this WITHOUT the originating conversation.
> Repo: `theislab/spatialdata-db-client` (public read SDK, `import sddb`, CLI `sddb-cli`, dist
> `spatialdata-db`). Local checkout `/Users/tim.treis/Documents/GitHub/spatialdata-db-client`.
> Companion scope doc (engine repo): `tasks/next-work-packages-2026-10-06.md` WP-A.

## 1. Goal

Make catalog discovery expressive and promote the gene index to a headline, consistent API — in ONE
`Catalog.query` entry point shared by Python, the `sddb-cli` CLI, and the MCP tools. Deliver numeric
range filters, composed facet+search+gene queries, and headline gene-API names, all additive and
backward-compatible.

## 2. Context — what already exists (do not rebuild)

Read these before implementing; the new work is narrow and sits on top of them.

- `src/sddb/catalog.py` — `Catalog.query(*, validation="pass", license_set=None, noncommercial=None,
  **facets) -> Results`. `**facets` ALREADY supports equality (`organism="human"`) and multi-value
  isin (`technology=["Xenium","VisiumHD"]` via list/tuple/set). Facet keys are validated against
  `schema.FACETS`. `Catalog.search(text) -> Results` does deterministic fuzzy/substring matching over
  facet values and carries a `matched` dict. `Catalog.genes` lazily builds a `GeneIndex`.
- `src/sddb/genes.py` — `GeneIndex.datasets_with(symbol, *, min_fraction=None, validation="pass") ->
  Results` and `GeneIndex.ranked(symbol, *, min_fraction=None, validation="pass") -> DataFrame`
  (per-dataset `uid, fraction_obs_detected, total_counts`, best detection first). `datasets_with`
  already delegates to `ranked` — ONE ranking path.
- `src/sddb/_catalog_schema.py` — the shared contract. `CATALOG_COLUMNS` (dtype map),
  `REQUIRED_COLUMNS`, `FACETS`, `GENE_INDEX_COLUMNS`, `validate(df, kind=...)`. Numeric columns in
  `CATALOG_COLUMNS`: `panel_size` (Int64), `n_obs` (Int64), `n_features` (Int64), `total_counts`
  (Float64), `size_bytes` (Int64).
- `src/sddb/cli.py` — Typer app. `query` command (facet flags + `--validation` + `--json`), `facets`,
  `genes` (uses `.ranked`), `cite` (HAND-ROLLS a query+search intersection by uid at lines ~198-201 —
  this is the duplication WP-A removes), `download`, `viewer-url`. Helpers `_facets`, `_validation`,
  `_print_table`, `_fmt`, `_suggest`.
- `src/sddb/mcp/tools.py` — `query_tool(catalog_url=None, validation="pass", **facets)` ALREADY
  forwards `**facets` to `Catalog.query`, so `col__op` range keys flow through unchanged.
  `genes_tool(symbol, ...)` calls `genes.datasets_with(symbol)`.
- `src/sddb/dataset.py` — `Results(df, *, matched=None, source=None)` is an ordered, sliceable
  `Sequence[Dataset]` with `.to_df/.download/.citations`. UNCHANGED by WP-A.
- `tests/_fixtures.py` — builds fixture catalog DataFrames/parquets for offline tests.

## 3. Premises corrected (scope is narrower than the one-line WP suggested)

1. **Multi-value facets already work** (isin). Nothing to build.
2. **The gene API is naming, not new compute** — `where_expressed` is `datasets_with` renamed;
   `ranked` already IS the per-dataset profile. Keep ONE ranking path.
3. **CLI `cite` already composes query+search** by uid-intersection — WP-A promotes that into
   `query(search=...)` and deletes the hand-rolled version (reuse/altitude win).

## 4. Design

### 4.1 `schema.RANGE_COLUMNS` (new, in `_catalog_schema.py`)

A tuple naming the numeric catalog columns that accept range operators — the shared contract so CLI
and MCP can validate identically:

```python
RANGE_COLUMNS: tuple[str, ...] = ("panel_size", "n_obs", "n_features", "total_counts", "size_bytes")
```

Derive intent, but write the literal tuple (explicit contract). It is exactly the `CATALOG_COLUMNS`
entries whose dtype is `Int64` or `Float64`.

### 4.2 `Catalog.query` — extended signature

```python
def query(
    self,
    *,
    validation: str | None = "pass",
    license_set: bool | None = None,
    noncommercial: bool | None = None,
    search: str | None = None,
    expressing: str | None = None,
    min_fraction: float | None = None,
    **facets: object,
) -> Results:
```

Behaviour (all new params optional; unset ⇒ byte-for-byte today's behaviour):

- **Range operators in `**facets`.** A key of the form `"<col>__<op>"` with `op ∈ {gte, gt, lte, lt}`
  is a range predicate on one of the numeric columns in `schema.RANGE_COLUMNS`. Split each `**facets`
  key on the LAST `__`; if the
  suffix is one of the four ops, it is a range key, else it is a plain facet key (so a plain facet
  name never collides — no current facet contains `__`). Collect range keys separately from equality
  facets.
  - The base column of a range key MUST be in `schema.RANGE_COLUMNS` AND present in the loaded
    catalog → else `ValueError(f"unknown range column {col!r}; valid: {list(schema.RANGE_COLUMNS)}")`.
  - Apply as `mask &= df[col] >= val` (etc.). Comparisons use pandas nullable dtypes; NA rows compare
    False (match the existing `mask.fillna(False)` discipline).
  - Plain (non-range) facet keys keep TODAY's validation: must be in `schema.FACETS`, else the current
    `ValueError(f"unknown facet(s) {bad}; valid facets: ...")`.
- **`search=`.** After the facet+range mask, intersect the surviving rows with
  `self.search(text)`'s uids. Carry `search()`'s `matched` dict onto the returned `Results`. If
  `search` matched nothing, the result is empty (search already warns).
- **`expressing=` (+ optional `min_fraction=`).** Intersect by uid with
  `self.genes.ranked(expressing, min_fraction=min_fraction, validation=validation)["uid"]`. First use
  downloads the gene index (document in docstring; ~142 MB, cached). A missing/corrupt gene index
  propagates the `ValueError` `GeneIndex` already raises.
- **`min_fraction` without `expressing`** → `ValueError("min_fraction requires expressing=<symbol>")`.
- Returns a single `Results` (already sliceable/downloadable/citeable).

Implementation note: factor a small private helper `_split_facets(facets) -> (equality, ranges)` and
keep `query` readable. Do not duplicate the existing equality/isin loop — extend it.

### 4.3 `GeneIndex` headline names

- Rename the method body to `where_expressed(self, symbol, *, min_fraction=None, validation="pass")
  -> Results` (identical behaviour to today's `datasets_with`).
- Keep `datasets_with = where_expressed` as a one-line alias assignment (so `mcp/tools.py` and any
  user code keep working; client is pre-PyPI, no deprecation warning needed).
- `ranked` stays as-is; update its docstring to name it the per-dataset detection **profile** (the
  WP's `profile(symbol)` idea — no separate method, YAGNI).

### 4.4 CLI (`cli.py`)

- `query` command gains options, all routed through the extended `Catalog.query` (no new logic in the
  CLI): `--expressing SYMBOL`, `--min-fraction FLOAT`, `--search TEXT`, `--min-obs INT`,
  `--min-features INT`. Map `--min-obs N` → `query(n_obs__gte=N)` and `--min-features N` →
  `query(n_features__gte=N)`. (total_counts/panel_size range flags deferred — add when asked.)
- `cite`: replace the hand-rolled query+search intersection (lines ~198-201) with
  `cat.query(validation=val, search=search, **facets)`. Same result, one code path.
- `genes` unchanged.

### 4.5 MCP (`mcp/tools.py`)

Add explicit passthrough params to `query_tool`: `expressing: str | None = None,
min_fraction: float | None = None, search: str | None = None`, forwarded to `Catalog.query`. Range
keys already flow via `**facets`. `genes_tool` keeps calling `datasets_with` (now an alias). Deeper
MCP expansion is WP-B — do not add new tools here.

## 5. Error handling (all `ValueError`, actionable message)

| Condition | Message shape |
|---|---|
| range key base col not numeric/unknown | `unknown range column 'foo'; valid: [...]` |
| plain facet key unknown | existing `unknown facet(s) [...]; valid facets: [...]` |
| `min_fraction` set, `expressing` unset | `min_fraction requires expressing=<symbol>` |
| gene index missing/corrupt on `expressing` | propagate `GeneIndex`'s existing `ValueError` |

## 6. Testing (offline, no network)

Extend `tests/_fixtures.py`: ensure the fixture catalog has varied numeric values across
`n_obs/n_features/total_counts/panel_size`, and add a small fixture `gene_index.parquet` (or reuse
`tests/test_genes.py`'s fixture) wired so `Catalog.genes` resolves locally.

New/extended tests (place in `tests/test_catalog.py`, `tests/test_genes.py`, `tests/test_cli.py`):

- range: `n_obs__gte`, `n_obs__gt`, `n_features__lte`, `total_counts__lt` each select the right rows.
- facet + range combined.
- `search=` intersects (facet AND text) and carries `matched`.
- `expressing=` intersects; `expressing=` + `min_fraction=` narrows further.
- all four combined in one `query(...)` call.
- errors: unknown range column; `min_fraction` without `expressing`; unknown plain facet still raises.
- `GeneIndex.where_expressed(sym)` equals `GeneIndex.datasets_with(sym)` (alias).
- CLI: `sddb-cli query --expressing EPCAM --min-obs 1000 --search liver` runs and lists rows; `cite`
  still produces identical output after the refactor.

## 7. Acceptance

- Range + multi-facet + combined (facet+search+gene) queries work offline on a fixture catalog.
- CLI, MCP and Python share ONE `Catalog.query` code path; `cite`'s duplication is gone.
- `GeneIndex` exposes `where_expressed` (headline) with `datasets_with` alias; one ranking path.
- Gene API documented; range contract in `schema.RANGE_COLUMNS`.
- `ruff` + `mypy --strict` clean; `sphinx-build` clean; all existing tests still pass.
- Backward compatible: every existing `query(...)`/`datasets_with(...)` call behaves unchanged.

## 8. Out of scope (recorded, not built)

- Fluent chainable `Results.filter/.search/.expressing` (we chose extend-`query()`).
- `total_counts`/`panel_size` CLI range flags.
- New MCP tools (WP-B).
- gene_index slimming / row-group pushdown (WP-C).
