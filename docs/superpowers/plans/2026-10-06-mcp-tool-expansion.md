# WP-B MCP Tool Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the client's MCP server a complete discovery→retrieval surface for agents: expose WP-A's richer query through the live server tool, and add `facets`/`cite`/`download` tools (download plans by default) reusing existing functions.

**Architecture:** One shared `_query` helper in `tools.py` feeds every tool (same WP-A filter surface, no duplication); the server is a thin shell registering live tools. `Results.download`/`Results.citations`/`cli.facets` logic reused. No new dependencies.

**Tech Stack:** Python 3.12+, pandas, fsspec/s3fs, platformdirs, typer (unused here), pytest. mypy --strict + ruff. Repo `theislab/spatialdata-db-client` (import `sddb`, MCP `sddb-mcp`).

**Spec:** `docs/superpowers/specs/2026-10-06-mcp-tool-expansion-design.md`

## Global Constraints

- No new dependencies (platformdirs/fsspec/s3fs/pandas/pyarrow already present).
- Tools are pure, JSON-serializable functions in `tools.py`; `server.py` stays a thin shell. Every tool unit-testable offline (no real network/S3).
- `download_tool` copies NOTHING by default (plan only); `download=True` writes ONLY under `_download_dir()` (`$SDDB_MCP_DOWNLOAD_DIR` else `platformdirs.user_cache_dir("sddb")/"mcp-downloads"`).
- `ruff check` (`uvx ruff check`, NOT `uv run ruff`), `uv run mypy --strict src`, full `uv run pytest -q`, and `uv run sphinx-build -b html docs docs/_build/html -q` all clean; delete `docs/_build` after. Ignore a trailing vitessce `PythonFinalizationError` at interpreter shutdown (harmless).
- Base off current `origin/main` (tip `b240ffa`, WP-A merged) in an isolated worktree. Stage only touched files; no AI-attribution in commits. Leave an untracked `uv.lock` unstaged.
- Mirror the live-tool kwarg shape the existing `query` wrapper uses (drop-None facet dict).

## Review Focus

1. **`download_tool` default must not copy bytes** (plan only) — an agent calling it must get sizes, not a multi-GB transfer. → Task 4.
2. **`download=True` writes only under the sandbox** (`$SDDB_MCP_DOWNLOAD_DIR`), never cwd/arbitrary path. → Task 4.
3. **`cite_tool` empty cohort → `{"bibtex": "", "n": 0}`**, not a crash/warning-to-stderr failure. → Task 3.
4. **`facets_tool` unknown field → `ValueError`** listing valid facets. → Task 2.
5. **Live `query` tool drops None range args** — `min_obs=None` must NOT forward `n_obs__gte=None` into `Catalog.query`. → Task 5.

---

### Task 1: `_query` shared helper + refactor `query_tool`

**Files:**
- Modify: `src/sddb/mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

**Interfaces:**
- Consumes: `Catalog.query(*, validation, expressing, min_fraction, search, **facets)` (WP-A).
- Produces: `_query(catalog_url, *, validation="pass", expressing=None, min_fraction=None, search=None, **facets) -> Results`; `query_tool` unchanged externally.

- [ ] **Step 1: Write the failing test** (append to `tests/test_mcp_tools.py`)

```python
def test__query_helper_filters(tmp_path):
    from tests._fixtures import write_fixture_catalog
    from sddb.mcp.tools import _query

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    res = _query(cat.as_uri(), organism="human")
    assert sorted(d.uid for d in res) == ["uid0001", "uid0002", "uid0005"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test__query_helper_filters -v`
Expected: FAIL (`cannot import name '_query'`).

- [ ] **Step 3: Add `_query` and refactor `query_tool`**

In `src/sddb/mcp/tools.py`, add after `_records`:

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
    """Shared filter path for the MCP tools: one Catalog.query call with the WP-A surface."""
    return Catalog(catalog_url).query(
        validation=None if validation == "all" else validation,
        expressing=expressing,
        min_fraction=min_fraction,
        search=search,
        **facets,
    )
```

Add the import `from sddb.dataset import Dataset, Results` (Results is new here). Refactor `query_tool` to delegate:

```python
def query_tool(
    catalog_url: str | None = None,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    **facets: Any,
) -> list[dict[str, Any]]:
    """Query the catalog by facets; at most ``MAX_ROWS`` rows (a trailing ``{"note": ...}`` marks truncation).

    ``validation`` filters ``validation_status`` (default ``"pass"``; ``"all"`` includes every row).
    ``**facets`` accepts equality, lists (isin) and ``col__gte``/``gt``/``lte``/``lt`` ranges. ``expressing``
    (+ optional ``min_fraction``) and ``search`` compose as in :meth:`Catalog.query`.
    """
    df = _query(
        catalog_url, validation=validation, expressing=expressing,
        min_fraction=min_fraction, search=search, **facets,
    ).to_df()
    rows = _records(df.head(MAX_ROWS))
    if len(df) > MAX_ROWS:
        rows.append({"note": f"truncated: showing {MAX_ROWS} of {len(df)} rows; narrow the query"})
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_tools.py -v && uv run mypy --strict src/sddb/mcp/tools.py && uvx ruff check src/sddb/mcp/tools.py`
Expected: PASS, clean (existing query_tool tests still pass).

- [ ] **Step 5: Commit**

```bash
git add src/sddb/mcp/tools.py tests/test_mcp_tools.py
git commit -m "refactor(mcp): shared _query helper behind query_tool"
```

---

### Task 2: `facets_tool`

**Files:**
- Modify: `src/sddb/mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

**Interfaces:**
- Consumes: `schema.FACETS`, `Catalog.to_df()`.
- Produces: `facets_tool(field: str | None = None, catalog_url: str | None = None) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
def test_facets_tool(tmp_path):
    from tests._fixtures import write_fixture_catalog
    from sddb.mcp.tools import facets_tool
    import pytest

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    cols = facets_tool(catalog_url=cat.as_uri())
    assert "organism" in cols and "technology" in cols
    assert facets_tool("organism", catalog_url=cat.as_uri()) == ["human", "mouse"]
    with pytest.raises(ValueError, match="unknown facet"):
        facets_tool("not_a_facet", catalog_url=cat.as_uri())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_facets_tool -v`
Expected: FAIL (`cannot import name 'facets_tool'`).

- [ ] **Step 3: Implement**

Add to `tools.py` (needs `from sddb import _catalog_schema as schema` at the top):

```python
def facets_tool(field: str | None = None, catalog_url: str | None = None) -> list[str]:
    """List facet columns present in the catalog, or the sorted distinct values of ``field``."""
    df = Catalog(catalog_url).to_df()
    if field is None:
        return [c for c in schema.FACETS if c in df.columns]
    if field not in schema.FACETS or field not in df.columns:
        raise ValueError(f"unknown facet {field!r}; valid facets: {list(schema.FACETS)}")
    return sorted(str(v) for v in df[field].dropna().unique())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_tools.py::test_facets_tool -v && uv run mypy --strict src/sddb/mcp/tools.py && uvx ruff check src/sddb/mcp/tools.py`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add src/sddb/mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat(mcp): facets_tool (facet columns / distinct values)"
```

---

### Task 3: `cite_tool`

**Files:**
- Modify: `src/sddb/mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

**Interfaces:**
- Consumes: `_query(...)` (Task 1); `Results.citations(path, *, bib_url=None)`; `sddb.citations.parse_bibtex`.
- Produces: `cite_tool(catalog_url=None, *, validation="pass", expressing=None, min_fraction=None, search=None, bib_url=None, **facets) -> dict` returning `{"bibtex": str, "n": int}`.

Fixture note: `tests/_fixtures.py` has `make_fixture_citations_bib()` (keys `Smith2023`/`Lee2022`/`Wong2021`, matching the fixture catalog `study_id`s). `write_citations` fetches `citations.bib` next to the catalog, so write that sibling file in the test.

- [ ] **Step 1: Write the failing test**

```python
def test_cite_tool(tmp_path):
    from tests._fixtures import write_fixture_catalog, make_fixture_citations_bib
    from sddb.mcp.tools import cite_tool

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    (tmp_path / "citations.bib").write_text(make_fixture_citations_bib(), encoding="utf-8")
    out = cite_tool(catalog_url=cat.as_uri(), organism="human")
    assert out["n"] >= 1
    assert "@" in out["bibtex"] and "Smith2023" in out["bibtex"]
    empty = cite_tool(catalog_url=cat.as_uri(), organism="nonexistent_xyz")
    assert empty == {"bibtex": "", "n": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_cite_tool -v`
Expected: FAIL (`cannot import name 'cite_tool'`).

- [ ] **Step 3: Implement**

Add to `tools.py` (needs `import tempfile`, `from pathlib import Path`, `from sddb.citations import parse_bibtex`):

```python
def cite_tool(
    catalog_url: str | None = None,
    *,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    bib_url: str | None = None,
    **facets: Any,
) -> dict[str, Any]:
    """Return BibTeX text + entry count for the studies of a filtered cohort (empty cohort -> n=0)."""
    res = _query(
        catalog_url, validation=validation, expressing=expressing,
        min_fraction=min_fraction, search=search, **facets,
    )
    if len(res) == 0:
        return {"bibtex": "", "n": 0}
    with tempfile.TemporaryDirectory() as d:
        path = res.citations(Path(d) / "c.bib", bib_url=bib_url)
        text = path.read_text(encoding="utf-8")
    return {"bibtex": text, "n": len(parse_bibtex(text))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_tools.py::test_cite_tool -v && uv run mypy --strict src/sddb/mcp/tools.py && uvx ruff check src/sddb/mcp/tools.py`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add src/sddb/mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat(mcp): cite_tool (BibTeX text for a filtered cohort)"
```

---

### Task 4: `manifest.plan_sizes` + `_download_dir` + `download_tool`

**Files:**
- Modify: `src/sddb/manifest.py` (add `plan_sizes`)
- Modify: `src/sddb/mcp/tools.py` (add `_download_dir`, `download_tool`)
- Test: `tests/test_manifest.py` (plan_sizes), `tests/test_mcp_tools.py` (download_tool)

**Interfaces:**
- Consumes: `_query(...)` (Task 1); `Results.download(dest, *, workers=4) -> Manifest`; `Manifest.entries` (list of `ManifestEntry` with `.uid/.size_bytes/.status`); `fsspec`, `sddb.remote.storage_options`; `platformdirs`.
- Produces: `manifest.plan_sizes(results) -> list[tuple[str, int]]`; `tools._download_dir() -> Path`; `tools.download_tool(catalog_url=None, *, download=False, workers=4, validation="pass", expressing=None, min_fraction=None, search=None, **facets) -> dict`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_manifest.py`:

```python
def test_plan_sizes_local_store(tmp_path):
    import pandas as pd
    from sddb.dataset import Results
    from sddb.manifest import plan_sizes
    from tests._fixtures import make_tiny_sdata_zarr

    store = make_tiny_sdata_zarr(tmp_path)
    res = Results(pd.DataFrame({"uid": ["uidX"], "zarr_url": [store.as_uri()]}))
    sizes = plan_sizes(res)
    assert len(sizes) == 1 and sizes[0][0] == "uidX" and sizes[0][1] > 0
```

In `tests/test_mcp_tools.py`:

```python
def test_download_tool_plan_no_copy(tmp_path, monkeypatch):
    from tests._fixtures import write_fixture_catalog
    import sddb.mcp.tools as t

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    monkeypatch.setattr(t, "plan_sizes", lambda res: [(d.uid, 1000) for d in res])
    called = {"download": False}
    monkeypatch.setattr(
        "sddb.dataset.Results.download",
        lambda self, *a, **k: called.__setitem__("download", True),
    )
    out = t.download_tool(catalog_url=cat.as_uri(), organism="human")
    assert called["download"] is False  # plan mode copies nothing
    assert out["datasets"] == 3 and out["total_bytes"] == 3000
    assert {p["uid"] for p in out["per_store"]} == {"uid0001", "uid0002", "uid0005"}


def test_download_tool_fetch_sandbox(tmp_path, monkeypatch):
    from tests._fixtures import write_fixture_catalog
    from sddb.manifest import Manifest, ManifestEntry
    import sddb.mcp.tools as t

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    sandbox = tmp_path / "dl"
    monkeypatch.setenv("SDDB_MCP_DOWNLOAD_DIR", str(sandbox))
    seen = {}

    def fake_download(self, dest, *, workers=4, **k):
        seen["dest"] = str(dest)
        return Manifest("now", None, [ManifestEntry("uid0001", "s3://x", str(dest), 5, "complete", None)])

    monkeypatch.setattr("sddb.dataset.Results.download", fake_download)
    out = t.download_tool(catalog_url=cat.as_uri(), organism="human", download=True)
    assert seen["dest"] == str(sandbox)  # fetched only into the sandbox
    assert out["dest"] == str(sandbox)
    assert out["entries"] == [{"uid": "uid0001", "size_bytes": 5, "status": "complete"}]
    assert "manifest_path" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifest.py::test_plan_sizes_local_store tests/test_mcp_tools.py -k download_tool -v`
Expected: FAIL (`plan_sizes` / `download_tool` undefined).

- [ ] **Step 3: Implement `plan_sizes`**

In `src/sddb/manifest.py`, add (module already imports `fsspec` and `from sddb.remote import storage_options`):

```python
def plan_sizes(results: Results) -> list[tuple[str, int]]:
    """Per-store ``(uid, total bytes)`` from a LIST/``du`` on each store — no data is copied."""
    out: list[tuple[str, int]] = []
    for ds in results:
        fs, root = fsspec.core.url_to_fs(ds.zarr_url, **storage_options(ds.zarr_url))
        out.append((ds.uid, int(fs.du(root))))
    return out
```

- [ ] **Step 4: Implement `_download_dir` + `download_tool`**

In `src/sddb/mcp/tools.py` (add `import os`, `from pathlib import Path`, `import platformdirs`, `from sddb.manifest import plan_sizes`):

```python
def _download_dir() -> Path:
    """Server-controlled download sandbox: $SDDB_MCP_DOWNLOAD_DIR or a platformdirs cache subdir."""
    env = os.environ.get("SDDB_MCP_DOWNLOAD_DIR")
    d = Path(env) if env else Path(platformdirs.user_cache_dir("sddb")) / "mcp-downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_tool(
    catalog_url: str | None = None,
    *,
    download: bool = False,
    workers: int = 4,
    validation: str = "pass",
    expressing: str | None = None,
    min_fraction: float | None = None,
    search: str | None = None,
    **facets: Any,
) -> dict[str, Any]:
    """Plan (default) or fetch a filtered cohort.

    ``download=False`` returns per-store + total sizes and copies nothing. ``download=True`` fetches
    into the server download dir and returns the manifest path.
    """
    res = _query(
        catalog_url, validation=validation, expressing=expressing,
        min_fraction=min_fraction, search=search, **facets,
    )
    if not download:
        sizes = plan_sizes(res)
        return {
            "datasets": len(sizes),
            "total_bytes": sum(b for _, b in sizes),
            "per_store": [{"uid": u, "bytes": b} for u, b in sizes],
            "note": "pass download=true to fetch into the server download dir",
        }
    dest = _download_dir()
    manifest = res.download(dest, workers=workers)
    return {
        "manifest_path": str(dest / "manifest.json"),
        "dest": str(dest),
        "entries": [{"uid": e.uid, "size_bytes": e.size_bytes, "status": e.status} for e in manifest.entries],
    }
```

Note: `Results.download` writes `manifest.json` into `dest`; the returned `manifest_path` points there.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py tests/test_mcp_tools.py -v && uv run mypy --strict src/sddb/manifest.py src/sddb/mcp/tools.py && uvx ruff check src/sddb/manifest.py src/sddb/mcp/tools.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/sddb/manifest.py src/sddb/mcp/tools.py tests/test_manifest.py tests/test_mcp_tools.py
git commit -m "feat(mcp): download_tool (plan by default, fetch to sandbox) + plan_sizes"
```

---

### Task 5: `server.py` live surface

**Files:**
- Modify: `src/sddb/mcp/server.py`
- Test: `tests/test_mcp_tools.py` (or `tests/test_mcp_server.py` if cleaner)

**Interfaces:**
- Consumes: `tools.query_tool/facets_tool/cite_tool/download_tool` (Tasks 1-4).
- Produces: live tools `query` (with expressing/min_fraction/search/min_obs/min_features), `facets`, `cite`, `download`; `describe`/`genes` unchanged.

- [ ] **Step 1: Write the failing test**

Register-and-forward test that avoids a real MCP transport. In `tests/test_mcp_tools.py`:

```python
def test_server_query_forwards_new_params(monkeypatch):
    import sddb.mcp.server as server
    import sddb.mcp.tools as tools

    seen = {}
    monkeypatch.setattr(tools, "query_tool", lambda **kw: seen.update(kw) or [])
    # capture the callables registered as tools
    registered = {}

    class FakeServer:
        def __init__(self, name): ...
        def tool(self, name, description=""):
            def deco(fn):
                registered[name] = fn
                return fn
            return deco
        def run(self): ...

    monkeypatch.setattr(server, "_server_class", lambda: FakeServer)
    server.build_server()
    assert {"query", "describe", "genes", "facets", "cite", "download"} <= set(registered)
    registered["query"](organism="human", expressing="EPCAM", search="liver", min_obs=1000)
    assert seen["expressing"] == "EPCAM" and seen["search"] == "liver"
    assert seen["n_obs__gte"] == 1000 and seen["organism"] == "human"
    assert "min_obs" not in seen and "n_features__gte" not in seen  # None range dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_server_query_forwards_new_params -v`
Expected: FAIL (new params not forwarded; `facets`/`cite`/`download` not registered).

- [ ] **Step 3: Implement**

In `src/sddb/mcp/server.py`, replace the inner `query` and add the three tools inside `build_server`:

```python
    def query(
        organism: str | None = None,
        tissue: str | None = None,
        disease: str | None = None,
        technology: str | None = None,
        assay: str | None = None,
        tier: str | None = None,
        validation: str = "pass",
        expressing: str | None = None,
        min_fraction: float | None = None,
        search: str | None = None,
        min_obs: int | None = None,
        min_features: int | None = None,
    ) -> list[dict[str, Any]]:
        facets = {"organism": organism, "tissue": tissue, "disease": disease}
        facets |= {"technology": technology, "assay": assay, "tier": tier}
        facets |= {"n_obs__gte": min_obs, "n_features__gte": min_features}
        return tools.query_tool(
            validation=validation, expressing=expressing, min_fraction=min_fraction, search=search,
            **{k: v for k, v in facets.items() if v is not None},
        )

    server.tool(name="query", description="Filter datasets by facets, ranges (min_obs/min_features), gene (expressing) and free-text search.")(query)

    def describe(uid: str) -> dict[str, Any]:
        return tools.describe_tool(uid)

    def genes(symbol: str) -> list[dict[str, Any]]:
        return tools.genes_tool(symbol)

    def facets(field: str | None = None) -> list[str]:
        return tools.facets_tool(field)

    def cite(
        organism: str | None = None, tissue: str | None = None, disease: str | None = None,
        technology: str | None = None, assay: str | None = None, tier: str | None = None,
        search: str | None = None, expressing: str | None = None, min_fraction: float | None = None,
        validation: str = "pass",
    ) -> dict[str, Any]:
        facets = {"organism": organism, "tissue": tissue, "disease": disease,
                  "technology": technology, "assay": assay, "tier": tier}
        return tools.cite_tool(
            validation=validation, search=search, expressing=expressing, min_fraction=min_fraction,
            **{k: v for k, v in facets.items() if v is not None},
        )

    def download(
        organism: str | None = None, tissue: str | None = None, disease: str | None = None,
        technology: str | None = None, assay: str | None = None, tier: str | None = None,
        search: str | None = None, expressing: str | None = None, min_fraction: float | None = None,
        validation: str = "pass", download: bool = False,
    ) -> dict[str, Any]:
        facets = {"organism": organism, "tissue": tissue, "disease": disease,
                  "technology": technology, "assay": assay, "tier": tier}
        return tools.download_tool(
            validation=validation, search=search, expressing=expressing, min_fraction=min_fraction,
            download=download, **{k: v for k, v in facets.items() if v is not None},
        )

    server.tool(name="describe", description="Catalog row and element shapes for one dataset uid.")(describe)
    server.tool(name="genes", description="Datasets whose gene index contains a gene symbol.")(genes)
    server.tool(name="facets", description="List facet columns, or the distinct values of a facet field.")(facets)
    server.tool(name="cite", description="BibTeX for the studies of a filtered cohort.")(cite)
    server.tool(name="download", description="Plan (sizes) or fetch a filtered cohort to the server download dir.")(download)
    return server
```

(Keep the existing `describe`/`genes` definitions — shown above for completeness; do not duplicate them.)

- [ ] **Step 4: Run tests + full gate**

Run:
```
uv run pytest -q
uv run mypy --strict src
uvx ruff check .
uv run sphinx-build -b html docs docs/_build/html -q && rm -rf docs/_build
```
Expected: all PASS/clean. (Ignore a trailing vitessce `PythonFinalizationError`.)

- [ ] **Step 5: Commit**

```bash
git add src/sddb/mcp/server.py tests/test_mcp_tools.py
git commit -m "feat(mcp): live query exposes expressing/search/ranges; register facets/cite/download"
```
