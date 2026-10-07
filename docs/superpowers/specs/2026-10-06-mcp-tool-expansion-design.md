# WP-B — MCP tool expansion (design)

> Self-contained: a fresh agent can execute this WITHOUT the originating conversation.
> Repo: `theislab/spatialdata-db-client` (public read SDK, `import sddb`, CLI `sddb-cli`, MCP `sddb-mcp`
> from the `[mcp]` extra). Local checkout `/Users/tim.treis/Documents/GitHub/spatialdata-db-client`.
> Base: branch off current `origin/main` (tip `b240ffa`, which has WP-A merged — PR #7).

## 1. Goal

Make the client's MCP server a complete discovery→retrieval surface for agents (the #1 audience):
expose WP-A's richer query through the *live* server tool, and add agent tools (`facets`, `cite`,
`download`) that reuse existing functions. Downloads are plan-by-default (show cost before copying
bytes). All tools stay pure functions behind a thin server, unit-testable offline.

## 2. Context — current state (do not rebuild)

`src/sddb/mcp/tools.py` (pure, JSON-serializable, no `mcp` import):
- `query_tool(catalog_url=None, validation="pass", expressing=None, min_fraction=None, search=None,
  **facets) -> list[dict]` — already forwards WP-A params to `Catalog.query`; caps at `MAX_ROWS=200`
  with a trailing `{"note": ...}` on truncation. `_records(df) -> list[dict]` helper.
- `describe_tool(uid, catalog_url=None)`, `genes_tool(symbol, catalog_url=None)`.

`src/sddb/mcp/server.py`:
- `build_server()` creates a FastMCP/MCPServer and registers live tools `query`, `describe`, `genes`.
- **GAP**: the live `query` tool hardcodes only facet params (organism/tissue/disease/technology/
  assay/tier/validation) — it does NOT expose WP-A's `expressing`/`min_fraction`/`search` or ranges,
  even though `query_tool` accepts them.

`src/sddb/catalog.py`: `Catalog.query(*, validation="pass", license_set=None, noncommercial=None,
search=None, expressing=None, min_fraction=None, **facets) -> Results` (WP-A). `<col>__gte/gt/lte/lt`
ranges on `schema.RANGE_COLUMNS`. `Catalog.to_df()`.

`src/sddb/dataset.py`: `Results.download(dest, *, workers=4, pin_versions=False,
allow_version_change=False) -> Manifest`; `Results.citations(path, *, bib_url=None) -> Path`;
`Results.to_df()`.

`src/sddb/manifest.py`: `Manifest`/`ManifestEntry` dataclasses (JSON via `dataclasses.asdict`);
`_copy_store(zarr_url, dest)` uses `fsspec.core.url_to_fs(zarr_url, **storage_options(zarr_url))`.
`src/sddb/remote.py`: `storage_options(url)` (anon S3 opts). `src/sddb/citations.py`:
`write_citations`, `parse_bibtex`. `src/sddb/cli.py`: `facets` command (facet columns, or distinct
values of a field) — the logic to mirror for `facets_tool`. `src/sddb/_catalog_schema.py`: `FACETS`.

Deps already present: `platformdirs`, `fsspec`, `s3fs`, `pyarrow`, `pandas`.

## 3. Design

### 3.1 `tools.py` — shared filter helper + new pure functions

Add one shared helper so every tool accepts the SAME WP-A filter surface (no duplication):

```python
def _query(
    catalog_url: str | None,
    *,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    **facets: Any,
) -> Results:
    return Catalog(catalog_url).query(
        validation=None if validation == "all" else validation,
        expressing=expressing, min_fraction=min_fraction, search=search, **facets,
    )
```

Refactor `query_tool` to build its DataFrame from `_query(...).to_df()` (no behaviour change; same
`MAX_ROWS` truncation + note).

**`facets_tool(field: str | None = None, catalog_url: str | None = None) -> list[str]`** — mirrors
`cli.facets`: with `field=None` return the facet columns present in the catalog (`[c for c in
schema.FACETS if c in df.columns]`); with a `field` return its sorted distinct non-null values; an
unknown `field` raises `ValueError` listing valid facets.

**`cite_tool(catalog_url=None, *, validation="pass", expressing=None, min_fraction=None, search=None,
bib_url=None, **facets) -> dict`** — filter via `_query(...)`; if empty return
`{"bibtex": "", "n": 0}`; else write via `Results.citations` to a `tempfile` (e.g.
`tempfile.TemporaryDirectory()/"c.bib"`), read it back as text, and return
`{"bibtex": "<text>", "n": <count via parse_bibtex>}`. (Agents want the BibTeX text, not a path; the
temp file is cleaned up — it is unrelated to the download sandbox.)

**`download_tool(catalog_url=None, *, download=False, workers=4, validation="pass", expressing=None,
min_fraction=None, search=None, **facets) -> dict`** — filter via `_query(...)`:
- `download=False` (default, a PLAN, copies nothing): return
  `{"datasets": N, "total_bytes": T, "per_store": [{"uid": .., "bytes": ..}, ...],
    "note": "pass download=true to fetch into the server download dir"}`. Sizes from a new
  `manifest.plan_sizes(results) -> list[tuple[str, int]]` (one `(uid, bytes)` per store). Empty
  cohort → `{"datasets": 0, "total_bytes": 0, "per_store": []}`.
- `download=True`: `manifest = results.download(_download_dir(), workers=workers)`; return
  `{"manifest_path": str, "dest": str, "entries": [{"uid", "size_bytes", "status"}...]}`.

**`_download_dir() -> Path`**: `Path(os.environ["SDDB_MCP_DOWNLOAD_DIR"])` if set, else
`Path(platformdirs.user_cache_dir("sddb")) / "mcp-downloads"`; `mkdir(parents=True, exist_ok=True)`.

### 3.2 `manifest.py` — `plan_sizes`

```python
def plan_sizes(results: Results) -> list[tuple[str, int]]:
    """Per-store (uid, total bytes) from a LIST/du on the remote store — no data copied."""
    out: list[tuple[str, int]] = []
    for ds in results:
        fs, root = fsspec.core.url_to_fs(ds.zarr_url, **storage_options(ds.zarr_url))
        out.append((ds.uid, int(fs.du(root))))
    return out
```
(`Results` is iterable of `Dataset`; `Dataset.uid`/`.zarr_url` exist. A missing store raises —
acceptable for a plan; `download=True`'s `Results.download` already records per-store failures.)

### 3.3 `server.py` — complete the live surface

- Extend the live `query` tool signature with `expressing: str | None = None`,
  `min_fraction: float | None = None`, `search: str | None = None`, `min_obs: int | None = None`,
  `min_features: int | None = None`; forward to `query_tool`, mapping `min_obs -> n_obs__gte` and
  `min_features -> n_features__gte` (drop None), i.e. the same mapping the CLI uses.
- Register three new live tools wrapping the pure functions, each with a one-line `description`:
  - `facets(field: str | None = None)` → `tools.facets_tool(field)`
  - `cite(organism=None, tissue=None, disease=None, technology=None, assay=None, tier=None,
    search=None, expressing=None, min_fraction=None, validation="pass")` → `tools.cite_tool(...)`
    (drop-None facets, same shape as the live `query` wrapper).
  - `download(<same filters as cite>, download=False)` → `tools.download_tool(...)`.
- Keep `describe`/`genes` unchanged.

## 4. Safety / constraints

- No network beyond the catalog + anon-S3 the client already uses. The plan path copies nothing
  (LIST/du only). `download=True` writes ONLY under `_download_dir()` (server-controlled).
- Tools return JSON (paths, sizes, text) — never stream bytes through the protocol.
- Pure-functions-behind-the-server split preserved → every tool unit-testable offline.
- No new dependencies. `ruff` + `mypy --strict` clean; full suite + sphinx clean.
- No AI-attribution in commits; stage only touched files.

## 5. Errors

| Condition | Behaviour |
|---|---|
| unknown facet / range / `facets` field | `ValueError` (from `Catalog.query` / `facets_tool`) |
| empty cohort in `cite_tool` | `{"bibtex": "", "n": 0}` (no crash) |
| empty cohort in `download_tool` | plan: `{"datasets": 0, ...}`; fetch: empty `entries` |
| a store size fails in plan | raises with the store url (plan is best-effort, no partial copy) |
| a store copy fails in `download=True` | recorded per-entry `status="failed"` (existing `Manifest` behaviour) |

## 6. Testing (offline, no network)

`tests/test_mcp_tools.py` (+ `tests/test_manifest.py` for `plan_sizes`):
- `facets_tool()` returns facet columns; `facets_tool("organism")` returns sorted values;
  `facets_tool("nope")` raises.
- `cite_tool(...)` on the fixture catalog returns BibTeX text + correct `n`; empty cohort → n=0.
- `download_tool(...)` PLAN: monkeypatch `manifest.plan_sizes` (or `fsspec`'s `du`) to return fixed
  sizes; assert the `{datasets,total_bytes,per_store}` shape and that NO download happened.
- `download_tool(..., download=True)`: point the catalog at a `file://` fixture store (reuse
  `tests/_fixtures.py::make_tiny_sdata_zarr`) OR monkeypatch `Results.download`; assert
  `manifest_path`/`entries` and that files land under a tmp `SDDB_MCP_DOWNLOAD_DIR` (set via
  monkeypatch/env) — never real S3.
- live wiring: `build_server()` registers `query/describe/genes/facets/cite/download`; the `query`
  wrapper forwards `expressing/search/min_obs` (assert via a patched `query_tool` capturing kwargs,
  or by calling the registered callable if the test MCP lib allows).
- `query_tool` behaviour unchanged after the `_query` refactor (existing tests still pass).

## 7. Acceptance

- Live MCP `query` tool exposes `expressing`/`min_fraction`/`search`/`min_obs`/`min_features`.
- `facets`, `cite`, `download` tools exist, return JSON, reuse `_query`/`Results`/`cli.facets` logic.
- `download_tool` plans by default (no copy) and fetches only to the sandbox on `download=True`.
- All tools offline-tested; `ruff` + `mypy --strict` + full suite + sphinx clean; no new deps.

## 8. Out of scope

- A `cohort_tool` (query_tool already returns the cohort; a summary can ride on query_tool later).
- Exposing every range column / upper bounds on the live tools (only min_obs/min_features, as CLI).
- Auth, remote lazy open, server-side catalog hosting, streaming bytes over MCP.
- gene_index slimming (WP-C).
