# WP-A Client Query + Gene API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add numeric range filters, composed facet+search+gene queries, and headline gene-API names to the `spatialdata-db` client, all through the single `Catalog.query` entry point, fully backward-compatible.

**Architecture:** Everything extends the existing `Catalog.query` + `GeneIndex`; no new classes, `Results` unchanged. `query` stays the one filter path that Python, CLI and MCP all call. New parameters are keyword-only and optional, so unset behaviour is byte-for-byte today's.

**Tech Stack:** Python 3.12+, pandas (nullable dtypes), pyarrow, typer (CLI), rapidfuzz (search), pytest. mypy --strict + ruff. Repo `theislab/spatialdata-db-client` (import `sddb`, CLI `sddb-cli`).

**Spec:** `docs/superpowers/specs/2026-10-06-client-query-gene-api-design.md`

## Global Constraints

- All new `query` params are keyword-only and optional; existing `query(...)`/`datasets_with(...)` calls behave unchanged.
- Client is pre-PyPI 0.x — no deprecation warnings needed for the `datasets_with` alias.
- All tests offline (no network); use `tests/_fixtures.py` helpers (`write_fixture_catalog`, `write_fixture_gene_index`) and `Catalog(path.as_uri(), cache_dir=tmp_path/"cache")`.
- `ruff check .`, `mypy --strict src`, and `sphinx-build` (docs) must stay clean; all existing tests must still pass.
- Base branch: off clean committed HEAD `f7d3a8c` (NOT the uncommitted collections work in the main checkout). Work in an isolated worktree.
- Stage only files each task touches; never `git add -A`. No AI-attribution lines in commits.

## Review Focus

1. **Range op on a non-numeric/unknown column** (`organism__gte=1`) → `ValueError` naming `RANGE_COLUMNS`, not a silent misread. → Task 1.
2. **Non-numeric range value** (`n_obs__gte="big"`) → `ValueError`, not a raw pandas `TypeError`. → Task 1.
3. **`min_fraction` without `expressing`** → `ValueError("min_fraction requires expressing=<symbol>")`. → Task 2.
4. **`expressing` a symbol absent from the index** → empty `Results`, no crash. → Task 2.
5. **`search=` matching nothing, combined with facets** → empty `Results`, no crash (search warns). → Task 2.

---

### Task 1: Numeric range filters in `Catalog.query`

**Files:**
- Modify: `src/sddb/_catalog_schema.py` (add `RANGE_COLUMNS` after `FACETS`)
- Modify: `src/sddb/catalog.py` (`query`, new `_split_facets` + range application)
- Test: `tests/test_catalog.py` (append)

**Interfaces:**
- Consumes: existing `Catalog.query(*, validation="pass", license_set=None, noncommercial=None, **facets) -> Results`; `schema.FACETS`, `schema.CATALOG_COLUMNS`.
- Produces: `schema.RANGE_COLUMNS: tuple[str, ...]`; `query` accepts `<col>__<op>` keys in `**facets` with `op ∈ {gte,gt,lte,lt}`.

Fixture facts (from `tests/_fixtures.py::make_fixture_catalog`): `n_obs` = uid0001:100000, uid0002:80000, uid0003:4000, uid0004:3500 (validation **fail**), uid0005:4200; `n_features` = 5000,5000,32000,32000,18000; `total_counts` = `n_obs*100`. Only uid0004 is `validation_status="fail"`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_catalog.py`:

```python
from tests._fixtures import write_fixture_catalog
from sddb import Catalog
import pytest


def _cat(tmp_path):
    p = write_fixture_catalog(tmp_path / "catalog.parquet")
    return Catalog(p.as_uri(), cache_dir=tmp_path / "cache")


def test_query_range_gte(tmp_path):
    cat = _cat(tmp_path)
    assert sorted(d.uid for d in cat.query(n_obs__gte=50_000)) == ["uid0001", "uid0002"]


def test_query_range_gt_excludes_boundary(tmp_path):
    cat = _cat(tmp_path)
    # n_obs > 4000 among validation=pass: uid0001, uid0002, uid0005 (uid0003=4000 excluded, uid0004 fails)
    assert sorted(d.uid for d in cat.query(n_obs__gt=4_000)) == ["uid0001", "uid0002", "uid0005"]


def test_query_range_lte_and_lt(tmp_path):
    cat = _cat(tmp_path)
    assert sorted(d.uid for d in cat.query(n_features__lte=5_000)) == ["uid0001", "uid0002"]
    assert sorted(d.uid for d in cat.query(total_counts__lt=500_000)) == ["uid0003", "uid0005"]


def test_query_facet_and_range(tmp_path):
    cat = _cat(tmp_path)
    assert sorted(d.uid for d in cat.query(organism="human", n_obs__gte=50_000)) == ["uid0001", "uid0002"]


def test_query_range_unknown_column_errors(tmp_path):
    cat = _cat(tmp_path)
    with pytest.raises(ValueError, match="range column"):
        cat.query(organism__gte=1)


def test_query_range_nonnumeric_value_errors(tmp_path):
    cat = _cat(tmp_path)
    with pytest.raises(ValueError):
        cat.query(n_obs__gte="big")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py -k range -v`
Expected: FAIL (query raises "unknown facet(s) ['n_obs__gte']" today — ranges not yet parsed).

- [ ] **Step 3: Add `RANGE_COLUMNS` to the schema**

In `src/sddb/_catalog_schema.py`, immediately after the `FACETS` tuple:

```python
# numeric columns that accept range operators (col__gte / __gt / __lte / __lt) in Catalog.query;
# exactly the CATALOG_COLUMNS entries whose dtype is Int64/Float64.
RANGE_COLUMNS: tuple[str, ...] = ("panel_size", "n_obs", "n_features", "total_counts", "size_bytes")
```

- [ ] **Step 4: Parse and apply ranges in `query`**

In `src/sddb/catalog.py`, add near the top (after imports):

```python
import operator

_RANGE_OPS = {"gte": operator.ge, "gt": operator.gt, "lte": operator.le, "lt": operator.lt}


def _split_facets(facets: dict[str, object]) -> tuple[dict[str, object], list[tuple[str, str, object]]]:
    """Separate plain equality facets from ``col__op`` range predicates (op in _RANGE_OPS)."""
    equality: dict[str, object] = {}
    ranges: list[tuple[str, str, object]] = []
    for key, val in facets.items():
        base, _, op = key.rpartition("__")
        if base and op in _RANGE_OPS:
            ranges.append((base, op, val))
        else:
            equality[key] = val
    return equality, ranges
```

Then in `query`, replace the `bad = [...]` / facet-loop region so equality facets are validated as today and ranges are applied separately. The body becomes:

```python
equality, ranges = _split_facets(facets)
bad = [k for k in equality if k not in schema.FACETS]
if bad:
    raise ValueError(f"unknown facet(s) {bad}; valid facets: {list(schema.FACETS)}")
bad_range = [c for c, _, _ in ranges if c not in schema.RANGE_COLUMNS or c not in df.columns]
if bad_range:
    raise ValueError(f"unknown range column(s) {bad_range}; valid: {list(schema.RANGE_COLUMNS)}")
df = self._df
mask = pd.Series(True, index=df.index)
if validation is not None:
    mask &= df["validation_status"] == validation
if license_set and "license_unknown" in df.columns:
    mask &= df["license_unknown"].fillna(True) == False  # noqa: E712
if noncommercial is not None and "license_noncommercial" in df.columns:
    mask &= df["license_noncommercial"].fillna(False) == noncommercial
for col, val in equality.items():
    if col not in df.columns:
        raise ValueError(f"facet column {col!r} is not present in this catalog (columns: {list(df.columns)})")
    mask &= df[col].isin(list(val) if isinstance(val, (list, tuple, set)) else [val])
for col, op, val in ranges:
    try:
        cmp = _RANGE_OPS[op](df[col], val)
    except TypeError as err:
        raise ValueError(f"range value for {col}__{op} must be numeric, got {val!r}") from err
    mask &= cmp.fillna(False).astype(bool)
return Results(df[mask.fillna(False).astype(bool)], source=self._source())
```

Note: compute `df = self._df` before the `bad_range` membership check needs `df.columns`; reorder so `df` is assigned first (move `df = self._df` above the `bad_range` line). Keep the method's existing docstring and extend it to describe the `col__op` range form.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -v && uv run ruff check src/sddb/catalog.py src/sddb/_catalog_schema.py && uv run mypy --strict src/sddb/catalog.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/sddb/_catalog_schema.py src/sddb/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): numeric range filters (col__gte/gt/lte/lt) in query"
```

---

### Task 2: Composed `search=` / `expressing=` / `min_fraction=` in `Catalog.query`

**Files:**
- Modify: `src/sddb/catalog.py` (`query` signature + composition)
- Test: `tests/test_catalog.py` (append)

**Interfaces:**
- Consumes: Task 1's `query` body; `self.search(text) -> Results` (carries `.matched`); `self.genes.ranked(symbol, *, min_fraction=None, validation="pass") -> DataFrame` with a `uid` column.
- Produces: `query(*, validation="pass", license_set=None, noncommercial=None, search=None, expressing=None, min_fraction=None, **facets) -> Results`.

Fixture facts (`make_fixture_gene_index`): EPCAM in uid0001 (fraction 0.4), uid0002 (0.3), uid0005 (0.25). `search("lung")` matches tissue `lung` → uid0001 only.

- [ ] **Step 1: Write the failing tests**

In `tests/test_catalog.py`:

```python
from tests._fixtures import write_fixture_gene_index


def _cat_genes(tmp_path):
    p = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    return Catalog(p.as_uri(), cache_dir=tmp_path / "cache")


def test_query_search_intersects(tmp_path):
    cat = _cat_genes(tmp_path)
    res = cat.query(organism="human", search="lung")
    assert [d.uid for d in res] == ["uid0001"]
    assert res.matched  # search's matched dict is carried through


def test_query_expressing(tmp_path):
    cat = _cat_genes(tmp_path)
    assert sorted(d.uid for d in cat.query(organism="human", expressing="EPCAM")) == ["uid0001", "uid0002", "uid0005"]


def test_query_expressing_min_fraction(tmp_path):
    cat = _cat_genes(tmp_path)
    assert [d.uid for d in cat.query(expressing="EPCAM", min_fraction=0.35)] == ["uid0001"]


def test_query_all_combined(tmp_path):
    cat = _cat_genes(tmp_path)
    res = cat.query(organism="human", n_obs__gte=50_000, expressing="EPCAM", search="lung")
    assert [d.uid for d in res] == ["uid0001"]


def test_query_min_fraction_without_expressing_errors(tmp_path):
    cat = _cat_genes(tmp_path)
    with pytest.raises(ValueError, match="min_fraction requires expressing"):
        cat.query(min_fraction=0.5)


def test_query_expressing_absent_symbol_empty(tmp_path):
    cat = _cat_genes(tmp_path)
    assert len(cat.query(expressing="NOPE")) == 0


def test_query_search_nothing_empty(tmp_path):
    cat = _cat_genes(tmp_path)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(cat.query(organism="human", search="zzzznotathing")) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py -k "search or expressing or combined or min_fraction" -v`
Expected: FAIL (query has no `search`/`expressing`/`min_fraction` params yet → TypeError).

- [ ] **Step 3: Extend the `query` signature and compose**

Change the signature to:

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

At the very top of the body (before building the mask), guard the misuse:

```python
if min_fraction is not None and expressing is None:
    raise ValueError("min_fraction requires expressing=<symbol>")
```

Build the base `Results` from facets+ranges exactly as Task 1 left it, but capture it in a variable instead of returning immediately; then intersect:

```python
result = Results(df[mask.fillna(False).astype(bool)], source=self._source())
matched: dict[str, list[str]] | None = None
if search is not None:
    hits = self.search(search)
    matched = hits.matched or None
    keep = set(hits.to_df()["uid"])
    rdf = result.to_df()
    result = Results(rdf[rdf["uid"].isin(keep)], matched=matched, source=self._source())
if expressing is not None:
    ranked = self.genes.ranked(expressing, min_fraction=min_fraction, validation=validation)
    keep = set(ranked["uid"])
    rdf = result.to_df()
    result = Results(rdf[rdf["uid"].isin(keep)], matched=matched, source=self._source())
return result
```

Extend the docstring: document `search`, `expressing`, `min_fraction`, and that `expressing` triggers a one-time gene-index download (~142 MB, cached).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -v && uv run mypy --strict src/sddb/catalog.py && uv run ruff check src/sddb/catalog.py`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add src/sddb/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): compose search/expressing/min_fraction in query"
```

---

### Task 3: `GeneIndex.where_expressed` headline name + `datasets_with` alias

**Files:**
- Modify: `src/sddb/genes.py` (rename method, add alias, docstring on `ranked`)
- Test: `tests/test_genes.py` (append)

**Interfaces:**
- Consumes: existing `GeneIndex.datasets_with(symbol, *, min_fraction=None, validation="pass") -> Results`.
- Produces: `GeneIndex.where_expressed(symbol, *, min_fraction=None, validation="pass") -> Results` (identical behaviour); `GeneIndex.datasets_with` remains as a one-line alias.

- [ ] **Step 1: Write the failing test**

In `tests/test_genes.py`:

```python
def test_where_expressed_is_datasets_with(tmp_path):
    cat = _cat(tmp_path)
    a = cat.genes.where_expressed("EPCAM").to_df()
    b = cat.genes.datasets_with("EPCAM").to_df()
    assert list(a["uid"]) == list(b["uid"])
    assert [d.uid for d in cat.genes.where_expressed("EPCAM", min_fraction=0.3)] == ["uid0001", "uid0002"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_genes.py::test_where_expressed_is_datasets_with -v`
Expected: FAIL (`GeneIndex` has no attribute `where_expressed`).

- [ ] **Step 3: Rename to `where_expressed` and alias**

In `src/sddb/genes.py`, rename the `datasets_with` method definition to `where_expressed` (body unchanged — it already delegates to `ranked`). Update its docstring first line to "Catalog datasets expressing ``symbol`` (case-insensitive), as a Results." After the method, inside the class body, add:

```python
    datasets_with = where_expressed  # backward-compatible alias (MCP + existing callers)
```

Also extend `ranked`'s docstring to name it the per-dataset detection **profile** for `symbol` (one row per catalog dataset, best detection first) — no behaviour change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_genes.py -v && uv run mypy --strict src/sddb/genes.py && uv run ruff check src/sddb/genes.py`
Expected: PASS, clean (existing `datasets_with` tests still pass via the alias).

- [ ] **Step 5: Commit**

```bash
git add src/sddb/genes.py tests/test_genes.py
git commit -m "feat(genes): where_expressed headline name, datasets_with alias"
```

---

### Task 4: CLI — compose flags on `query`, dedup `cite`

**Files:**
- Modify: `src/sddb/cli.py` (`query` options, `cite` refactor)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: Task 1+2 `Catalog.query` with `search`/`expressing`/`min_fraction`/`<col>__op`.
- Produces: `sddb-cli query` gains `--expressing/--min-fraction/--search/--min-obs/--min-features`; `cite` uses `query(search=...)` instead of its hand-rolled intersection.

- [ ] **Step 1: Write the failing test**

In `tests/test_cli.py` (follow the existing `CliRunner`/`write_fixture_catalog` pattern already in the file — load it and mirror it):

```python
def test_query_expressing_and_range_flags(tmp_path):
    from typer.testing import CliRunner
    from tests._fixtures import write_fixture_catalog, write_fixture_gene_index
    from sddb.cli import app

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    r = CliRunner().invoke(
        app,
        ["query", "--catalog", cat.as_uri(), "--expressing", "EPCAM", "--min-obs", "50000", "--search", "lung"],
    )
    assert r.exit_code == 0, r.output
    assert "uid0001" in r.output
    assert "uid0002" not in r.output  # excluded by search=lung
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_query_expressing_and_range_flags -v`
Expected: FAIL (`query` has no `--expressing`/`--min-obs`/`--search` options).

- [ ] **Step 3: Add the flags and route through `query`**

In `cli.py` `query` command, add options:

```python
    expressing: Annotated[str | None, typer.Option(help="Keep datasets expressing this gene symbol (downloads the gene index once).")] = None,
    min_fraction: Annotated[float | None, typer.Option(help="With --expressing: min fraction_obs_detected.")] = None,
    search: Annotated[str | None, typer.Option(help="Free-text search over facet values.")] = None,
    min_obs: Annotated[int | None, typer.Option(help="Keep datasets with n_obs >= this.")] = None,
    min_features: Annotated[int | None, typer.Option(help="Keep datasets with n_features >= this.")] = None,
```

Build the range kwargs and pass everything through:

```python
    ranges = {k: v for k, v in {"n_obs__gte": min_obs, "n_features__gte": min_features}.items() if v is not None}
    try:
        cat = Catalog(catalog)
        res = cat.query(validation=val, search=search, expressing=expressing, min_fraction=min_fraction, **facets, **ranges)
    except Exception as err:
        raise _fail(err) from err
```

(Keep the existing `_suggest` empty-result hint, JSON branch, metadata header, table print.)

In the `cite` command, replace the manual intersection block (the `if search is not None:` block that fetches `cat.search(search)` and filters by uid) with a single call:

```python
    try:
        cat = Catalog(catalog)
        res = cat.query(validation=val, search=search, **facets)
    except Exception as err:
        raise _fail(err) from err
```

(Delete the now-unused `Results` import if nothing else in `cli.py` uses it — check `_resolve`, which still constructs `Results`, so keep the import.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py tests/test_cli_cite.py -v && uv run mypy --strict src/sddb/cli.py && uv run ruff check src/sddb/cli.py`
Expected: PASS (existing `cite` tests still pass — same output, one code path).

- [ ] **Step 5: Commit**

```bash
git add src/sddb/cli.py tests/test_cli.py
git commit -m "feat(cli): query --expressing/--search/--min-obs/--min-features; cite reuses query"
```

---

### Task 5: MCP — `query_tool` composition passthrough

**Files:**
- Modify: `src/sddb/mcp/tools.py` (`query_tool` params)
- Test: `tests/test_mcp_tools.py` (append)

**Interfaces:**
- Consumes: Task 1+2 `Catalog.query`.
- Produces: `query_tool(catalog_url=None, validation="pass", expressing=None, min_fraction=None, search=None, **facets)` — range keys already flow via `**facets`.

- [ ] **Step 1: Write the failing test**

In `tests/test_mcp_tools.py` (mirror the file's existing fixture-catalog pattern):

```python
def test_query_tool_expressing_and_range(tmp_path):
    from tests._fixtures import write_fixture_catalog, write_fixture_gene_index
    from sddb.mcp.tools import query_tool

    cat = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    rows = query_tool(cat.as_uri(), expressing="EPCAM", n_obs__gte=50_000)
    uids = {r["uid"] for r in rows if "uid" in r}
    assert uids == {"uid0001", "uid0002"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_query_tool_expressing_and_range -v`
Expected: FAIL (`query_tool` has no `expressing` param → TypeError).

- [ ] **Step 3: Add passthrough params**

In `src/sddb/mcp/tools.py`, change `query_tool` to:

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
    df = (
        Catalog(catalog_url)
        .query(
            validation=None if validation == "all" else validation,
            expressing=expressing,
            min_fraction=min_fraction,
            search=search,
            **facets,
        )
        .to_df()
    )
    rows = _records(df.head(MAX_ROWS))
    if len(df) > MAX_ROWS:
        rows.append({"note": f"truncated: showing {MAX_ROWS} of {len(df)} rows; narrow the query"})
    return rows
```

- [ ] **Step 4: Run tests to verify they pass + full suite**

Run: `uv run pytest -q && uv run mypy --strict src && uv run ruff check .`
Expected: PASS, clean. Also build docs: `uv run sphinx-build -b html docs docs/_build/html -q` (expect clean; delete `docs/_build` after).

- [ ] **Step 5: Commit**

```bash
git add src/sddb/mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat(mcp): query_tool passes expressing/min_fraction/search through"
```
