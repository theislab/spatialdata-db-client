# spatialdata-db-client — design spec

- **Date:** 2026-10-05
- **Status:** draft for review
- **Repo:** currently `theislab/spatialdata-db-client` (empty, private); public before release
- **Related:** `REQUIREMENTS.md` §2.4 (API-01…07), `specs/qc_and_viz.md` §6 (CAT-05 index),
  `docs/superpowers/specs/2026-10-01-spatialdata-db-website-design.md`

## 1. Vision

A lean, pip-installable Python package that a researcher who has **never heard of us and never
heard of lamin** can `pip install` and immediately get more out of the spatialdata-db collection
than the website gives them: **filter the whole catalog in pandas** (the hero capability), load
any dataset as a `SpatialData` object, open one gene column or image region from a 20 GB store
without downloading it, fetch a reproducible cohort with a resumable manifest, cite it, and view
it interactively — all **anonymously, no account, lamin never mentioned**.

It is the **canonical programmatic access layer** for the collection: the website is one view over
the same catalog contract, and this engine repo consumes the same layer internally. One set of
tools serves outside consumers *and* our own agent sessions.

**Audience priority** (last to be sacrificed, in order): our own agent sessions → tool/pipeline
builders → bench/analysis biologists → ML/computational researchers. This ranking is why the
tool/MCP surface and a stable catalog contract rank above heavy ML affordances.

## 2. Non-goals

- No writing, curation, conversion, discovery, registration, or ingestion — those stay in the
  engine.
- No lamin, no accounts, no credentials in the consumer path. No `bionty`, `spatialdata-io`,
  `spatialdata-plot` dependency; no `vitessce` in the base install.
- No opening of arbitrary zarr stores — **catalog-resolved datasets only** (keeps the contract
  tight; a general zarr opener is a different tool).
- No ML/dataloader utilities in v1 (out of scope; may return later as recipe notebooks).
- No bundled LLM weights, no load-bearing hosted-inference dependency.
- No telemetry of any kind, ever.
- No web server / hosted service. A thin CLI and an optional MCP server are in scope; a service is
  not.

## 3. Key decisions

| # | Decision |
|---|---|
| D61 | Consumer query + load must not touch the lamin metadata DB. Query a published static catalog; load direct from anon S3. lamindb is not a consumer dep. (Evidence below.) |
| D62 | The **client takes the public name**: PyPI `spatialdata-db`, import `sddb`. The **engine is renamed** (package + its `sddb` CLI) as a separate migration of this repo. |
| D63 | **MCP server ships in v1** as an optional `[mcp]` extra (agents are the #1 audience). |
| D64 | Interactive viz is an optional `[viz]` extra that **reuses the Vitessce configs we already publish** (+ `easy_vitessce`); the base install stays viz-dependency-free. |
| — | License **BSD-3-Clause**; develop under **theislab**, donate to **scverse** once listed/stable; **scverse ecosystem listing targeted for v1**. |

**D61 evidence** (probe, 2026-10-05, from a laptop on a normal network): anonymous S3 data access
works (LIST + GET → HTTP 200, no credentials); the lamin metadata DB is unreachable
(`.using("scverse/spatialdata-db")` opens a *direct TCP connection to the RDS Postgres* at
`…rds.amazonaws.com:5432`, firewalled off-HPC, times out). Since lamindb resolves both `filter(...)`
*and* `get(uid).load()` through that DB, neither works for an off-HPC user — contradicting API-01
("query through lamindb") given API-06 ("no account"). Catalog-first resolves it and matches CAT-05.
Verification task: confirm the RDS security group is intentionally IP-restricted.

## 4. Architecture

```
HPC (can reach RDS)                         Anywhere (laptop, CI, Colab)
─────────────────────                       ────────────────────────────
nightly inventory job                       spatialdata-db  (import sddb)
  queries prod lamin        ── publish ──>   catalog.parquet  (anon HTTPS, website/Cloudflare)
  (engine; uses the client's                     │ fetch + cache  (latest + dated snapshots)
   catalog schema)                                │ query() / search()  -> Results (pandas)
                                                   ▼
                                       uid -> direct s3:// zarr URL (from catalog)
                                                   │
                                                   ▼
                              spatialdata.read_zarr(url)  (anon S3 via fsspec/s3fs; lazy by default)
```

Dependency direction is strictly **engine → client**. The client is the base read SDK; the engine
pip-depends on it and adopts its `Catalog` / `open_sdata` in the inventory generator, `load-check`,
and QC-open. The client never imports the engine or lamin.

## 5. Public API

`Catalog` object returning `Dataset` records; all returns are plain pandas or `SpatialData`.

```python
import sddb

cat = sddb.Catalog()                            # fetch + cache catalog.parquet (anon HTTPS)
cat = sddb.Catalog(version="2026-09-01")        # pin a dated snapshot (reproducibility)
cat = sddb.Catalog(cache_dir="/data/sddb")      # arg > SDDB_CACHE_DIR env > platformdirs default

# ---- query (the hero) ----
res = cat.query(organism="human", assay="10x Xenium", validation="pass")  # -> Results
df  = res.to_df()                               # underlying pandas DataFrame
len(res); res[0]                                # sequence-like

# ---- search (deterministic NL, offline) ----
res = cat.search("human breast cancer xenium")  # fuzzy/synonym over enumerated vocab -> Results

# ---- a single dataset ----
d = res[0]
d.uid, d.assay, d.organism, d.zarr_url, d.size_bytes
d.elements()                                    # remote introspection, no full download
sdata = d.load()                                # lazy by default (dask-backed over S3)
sdata = d.load(lazy=False)                       # cache + full load
sdata = d.load(version="...")                    # pin a specific dataset version (default: latest)
url = d.viewer_url()                            # vitessce.io link (reuses our shipped config)
d.view()                                        # open that link in a browser (stdlib)

# ---- cohorts ----
m = res.download("./data", workers=4)           # thread-pool, resumable, writes manifest.json
res.citations("refs.bib")                       # filter published citations.bib to this cohort
m = sddb.Manifest.read("./data/manifest.json")  # .refetch(dest): re-download exactly the recorded stores

# ---- cross-dataset gene search ----
hits = cat.genes.datasets_with("EPCAM")         # from gene_index.parquet -> Results
```

Notes:
- `query()` defaults `validation="pass"` (API-05). `validation`/`tier` are parsed into the catalog
  at generation time (not lamin-queryable features), so the client filters them as ordinary columns.
- **Load/fetch latest by default; pinning is opt-in** (`version=` on `load`). Manifests record the
  **catalog** version when `pin_versions=True`; a later `download` into the same dest against a
  different catalog version raises `ManifestVersionMismatch` unless `allow_version_change=True`.
  `Manifest.refetch(dest)` replays the recorded URLs. Per-dataset byte-level pinning across
  republishes is future work.
- Offline: if the catalog host is unreachable, fall back to the last-cached catalog with a warning
  naming its age; error only if no cache exists.

## 6. Catalog contract

`catalog.parquet` — one row per published dataset (latest version), keyed by public `uid`. This is
CAT-05 (`fleet/index.parquet`), finally published. **Parquet** (columnar, typed, tiny, read natively
by pandas/arrow/polars/R — supports the deferred R path). Column groups (authoritative list in
`specs/qc_and_viz.md` §6):

- **Identity:** `uid`, `canonical_key`, `study_id`, `collection_name`, `issue_url`
- **Biology:** `organism`, `tissue`, `disease`, `development_stage` (+ ontology IDs)
- **Technology:** `technology`, `technology_version`, `panel_size`
- **Flags + validation:** every PLAYBOOK §6.2 flag, plus `validation_status`, `tier`
- **License:** `license_spdx`, `license_text`, `license_unknown`
- **Stats:** headline stats (`qc_and_viz` §1.1) + per-family keys
- **Access:** `zarr_url` (direct `s3://`), `card_url`, `vitessce_url`, `size_bytes`
- **Provenance:** `container_tag`, `sddb_version`, `published_at`

Companions: `gene_index.parquet` (`symbol`, `feature_id`, `uid`, `total_counts`,
`fraction_obs_detected`) and `citations.bib` (one entry per study).

- **Producer:** the engine's nightly inventory job (runs on HPC, reaches the DB). New requirement
  the client imposes: the catalog **must carry the direct `s3://` `zarr_url`** (today's
  `inventory.json` lacks it; CAT-05 specifies it). The client owns `_catalog_schema.py` (columns +
  dtypes); the inventory generator imports it, so producer and consumer cannot drift (contract test).
- **Host:** published to the website (Cloudflare) in the same nightly deploy as `inventory.json`,
  at a stable URL under `spatialdata-db.com` (exact path set at implementation). Publish **latest +
  immutable dated snapshots** so `Catalog(version=...)` reproduces a past state.
- **Client cache:** platformdirs user cache by default (matches lamindb), overridable by arg >
  `SDDB_CACHE_DIR` env > default; `ETag`/`Last-Modified` conditional refresh; `refresh=True` forces.

## 7. Module layout

```
src/sddb/
  __init__.py          # re-exports Catalog, Dataset, Results, Manifest, open_sdata
  catalog.py           # fetch+cache catalog.parquet, query(), search() -> Results
  dataset.py           # Dataset / Results records: load(), elements(), viewer_url(), view(), download(), citations()
  remote.py            # anon S3 (fsspec/s3fs) resolve + read_zarr + remote element introspection
  manifest.py          # cohort manifest write/read + thread-pool resumable fetch (version pinning)
  genes.py             # gene_index reader (cross-dataset gene search)
  search.py            # deterministic NL -> structured query (rapidfuzz + synonyms over the vocab)
  citations.py         # filter published citations.bib to a cohort
  _catalog_schema.py   # THE shared column contract (also imported by the engine's inventory gen)
  _cache.py            # cache dir resolution + conditional HTTP fetch + offline fallback
  cli.py               # thin CLI (base): sddb query / download / viewer-url
  mcp/                 # [mcp] extra: MCP server exposing query/open/describe/genes as tools
  viz.py               # [viz] extra: in-notebook interactive via shipped config + easy_vitessce
```

## 8. Backend / consumer boundary and internal overlap

| Tool (in client) | Consumer benefit over website | Internal reuse (engine) |
|---|---|---|
| `Catalog.query/search` | filter thousands in seconds; reproducible cohorts | replaces ad-hoc `gh`/lamin scripts for "which datasets need X" |
| `open_sdata(uid, lazy)` | load any dataset in one line, no creds/HPC | shared with `load-check` + QC-open; debug a *published* artifact from a laptop |
| remote `elements()` | inspect a 5 GB store before committing | convert/publish triage |
| `Manifest` + resumable fetch | reproducible cohort pulls | batch convert/publish cohorts stop re-rolling this |
| `viewer_url` / `[viz]` | instant scriptable / in-notebook interactive view | logic migrates out of the web worktree |
| gene index | cross-dataset gene search | curation QC |
| MCP tools | agents drive the collection directly | our Claude sessions call the same tools |

First engine adopter: the **inventory/catalog generator** (already queries prod read-only) becomes
the catalog producer against the shared `_catalog_schema.py`.

## 9. Interfaces

- **CLI (base):** `sddb query`, `sddb download`, `sddb viewer-url` (typer; no heavy deps). Note the
  **brand overlap with the engine's existing `sddb` CLI** — resolved by D62 renaming the engine's
  console script so the public `sddb` command belongs to the client.
- **MCP server (`[mcp]` extra, v1):** exposes `catalog.query`, `open`, `describe`, `genes` as tools.
  User brings their own model; we ship the tool, not a model. Serves the #1 audience and our own
  sessions from day one.
- **Interactive viz (`[viz]` extra):** render the per-dataset Vitessce config we already publish via
  the `vitessce` widget, and/or `.view_interactive()` on a loaded `SpatialData` via `easy_vitessce`
  (MIT, Vitessce team, PyPI). Base install unaffected.
- **NL search:** deterministic `search()` in v1; an optional bring-your-own-endpoint `[llm]` parser
  is a later extra (never bundled, never default).

## 10. Dependencies & scverse conventions

**Runtime (lean base):** `spatialdata`, `pandas>=2.2`, `pyarrow`, `fsspec`, `s3fs`, `rapidfuzz`.
Extras: `[viz]` (`vitessce`, `easy_vitessce`), `[mcp]` (MCP SDK), `[llm]` (later). Returns/accepts
`SpatialData` — satisfies the mandatory scverse data-structure criterion.

**Packaging (current scverse `main`):**
- Build: `hatchling` + **hatch-vcs** dynamic versioning (`dynamic = ["version"]`,
  `[tool.hatch.version] source = "vcs"`); `src`-layout.
- `requires-python = ">=3.12"`; classifiers 3.12/3.13/3.14; SPEC 0 lower-bound pins, no EOL pins.
- **ruff** (line-length 120, numpy docstring convention, rule set matching spatialdata:
  `B,C4,D,E,F,I,UP,W,NPY,PT,RET,TID`), **pre-commit** + pre-commit.ci, `pyproject-fmt`, `zizmor`.
- **mypy** strict (`disallow_untyped_defs`, `disallow_incomplete_defs`).
- Docstrings numpydoc; docs **Sphinx + sphinx-book-theme + myst-nb + scanpydoc** on Read the Docs;
  API autosummary; MyST/notebook tutorials.
- **pytest** (`--import-mode=importlib`), **coverage** scoped to the package, GitHub Actions with a
  hatch-generated env matrix + a pre-release-deps leg, **Codecov** (`use_oidc`), `alls-green` gate.
- **BSD-3-Clause** + Code of Conduct; bootstrap from cookiecutter-scverse; submit `meta.yaml` to
  `scverse/ecosystem-packages` at v1. Channels: **PyPI + conda-forge** (feedstock).
- **API stability:** 0.x may change (documented public surface + semver discipline), freeze at 1.0.

## 11. Error handling

- **Offline / fetch fails:** cached catalog + warning with its age; error only if no cache.
- **Stale catalog:** `generated_at` exposed; conditional refresh on `ETag`.
- **Unknown/dropped uid:** explicit error naming the uid + catalog version.
- **Remote open failure:** wrap with uid + `zarr_url`; distinguish "not found" from "transport error".
- **Resumable download:** manifest records per-file status + checksum; re-run skips completed files;
  partials re-fetched, not appended; a pinned manifest verifies the catalog version matches.
- **`search()` low confidence:** return best-effort matches *and* what it matched on; never silently
  return everything.

## 12. Testing strategy

- **Unit** over a committed **fixture catalog** (synthetic rows): query, search, manifest, schema,
  offline cache, citations. No network; the bulk of the suite.
- **Integration (opt-in, `@pytest.mark.network`):** against real public S3 + a real catalog on the
  smallest dataset — prove anon `elements()` + lazy `load()` transfer well under the full store
  (API-02: < 5% for one table) and that a resumable download resumes without re-fetch (API-03).
- **Contract test:** engine inventory generator + client both import `_catalog_schema.py`; assert a
  generated catalog validates against it (prevents drift).
- **Tutorials** (myst-nb) double as end-to-end checks on public data.

## 13. v1 scope vs later

**v1:** `Catalog.query` + deterministic `search`, `open_sdata` lazy/full, remote `elements()`,
`Manifest` + resumable `download`, `citations`, `viewer_url` + `view`, gene index, thin CLI,
`[mcp]` extra, `[viz]` extra, dated catalog snapshots + version pinning.

**Later:** `[llm]` BYO-endpoint parser; dataloader recipe notebooks; R access (catalog kept
R-friendly); donate repo to scverse; bioconda.

## 14. Open questions / risks

- **Catalog publishing not yet wired** (CAT-05 missing): the nightly job must emit `catalog.parquet`
  (with `zarr_url`) + `gene_index.parquet` + dated snapshots to the website host. Critical-path; the
  client is inert without it.
- **Engine rename (D62)** is a non-trivial migration of *this* repo (package import, `sddb` console
  script, docs, downstream refs). It is a separate plan from the client build; sequence so the public
  `sddb` command/name is free before the client's public release.
- **Exact catalog URL** under `spatialdata-db.com` — finalized with the website deploy.
- **RDS security-group policy** — verify intentionally IP-restricted.
- **Repo visibility + license + CoC** — flip public, add BSD-3 + CoC before release.

## 15. Rollout (high level; detailed plan via writing-plans)

1. Scaffold from cookiecutter-scverse; `_catalog_schema.py`; CI/docs/pre-commit/BSD-3.
2. `remote.py` + `open_sdata` + `elements()` against public S3 (integration-proven).
3. `catalog.py` (`Catalog`, `query`, `Results`, `Dataset`) over a fixture catalog; dated snapshots +
   version pinning.
4. Wire the engine inventory generator to emit `catalog.parquet` (with `zarr_url`) + snapshots
   against the shared schema, published nightly. (Unblocks everything.)
5. `manifest.py` resumable thread-pool download; `genes.py`; `citations.py`; `viewer_url`/`view`.
6. `search.py` deterministic NL; thin CLI.
7. `[mcp]` extra (server + tool schema); `[viz]` extra (shipped config + easy_vitessce).
8. Engine adopts the client in `load-check` + QC-open.
9. Tutorials, Read the Docs, PyPI + conda-forge release, scverse ecosystem submission.
10. Separately: engine rename (D62). Later: `[llm]`, dataloader notebooks, R, scverse donation.
