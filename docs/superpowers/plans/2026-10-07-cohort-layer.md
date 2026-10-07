# Cohort Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the client-side reproducible-cohort layer — `SpatialDataCohort` → `freeze()` → `FrozenCohort` → `split()`/`check()`/`build()` — so a catalog query becomes a citable, reconstructable scientific artifact.

**Architecture:** Thin return object + focused delegating modules (the existing `citations.py`/`manifest.py` idiom). Two types: `SpatialDataCohort` (live, catalog-bound) and `FrozenCohort` (immutable, self-contained, hashed, with enforced verify-on-read). New behavior lives in small modules (`cohort.py`, `objectmeta.py`, `freeze.py`, `concat.py`, `split.py`, `adapter.py`); `Results` is renamed outright (pre-release, no alias).

**Tech Stack:** Python ≥3.12, pandas, anndata, zarr, fsspec, spatialdata (read-only), pytest. mypy-strict + ruff.

**Spec:** `docs/superpowers/specs/2026-10-07-cohort-layer-design.md` — read it alongside this plan.

## Global Constraints

- Python **≥3.12**; `from __future__ import annotations` at the top of every new module (matches the repo).
- No new runtime dependencies — anndata/zarr/fsspec/spatialdata are already declared.
- Public import stays `import sddb`; CLI `sddb-cli`; dist `spatialdata-db`.
- mypy-strict clean, ruff clean, `sphinx-build` clean. Network tests marked `@pytest.mark.network` (deselected by default).
- Metadata reads NEVER pull array chunks (reuse `sddb.remote.elements` / `storage_options`).
- `Results` is **deleted**, not aliased.

## Review Focus

Inputs/conditions the spec implies but that a task's happy-path tests might miss — each is pinned to a test in the owning task:

- **A store republished at the same URL after freeze** → a `FrozenCohort` data read must raise `CohortDriftError`, not silently return changed data. (Task 6, verify-on-read test.)
- **One member's store unreachable at freeze** → `freeze()` raises and names the uid; no partial cohort. (Task 5, all-or-nothing test.)
- **Cohort with fewer groups than non-zero split folds** → `split()` raises; never emits an empty val/test. (Task 8, small-cohort test.)
- **Same split requested on a different machine** → identical assignment (sorted keys + seeded `default_rng`), not pandas/`set`-ordering-dependent. (Task 8, determinism test.)
- **`modalities()` on a live cohort** → reads catalog flags only, performs no network I/O. (Task 2, no-network test.)

---

## Task 1: Rename `Results` → `SpatialDataCohort`; move to `cohort.py`

**Files:**
- Create: `src/sddb/cohort.py`
- Modify: `src/sddb/dataset.py` (remove `Results`), `src/sddb/catalog.py`, `src/sddb/citations.py`, `src/sddb/manifest.py`, `src/sddb/mcp/tools.py`, `src/sddb/cli.py`, `src/sddb/__init__.py`
- Test: all existing tests (renamed references)

**Interfaces:**
- Produces: `SpatialDataCohort(df: pd.DataFrame, *, matched: dict[str,list[str]] | None = None, source: Source | None = None)` — same constructor and `Sequence[Dataset]` behavior `Results` had, plus `to_df`, `citations`, `citations_text`, `download`. `_df`, `matched`, `_source` attributes preserved.

- [ ] **Step 1: Move the class.** Cut the `Results` class body out of `dataset.py` into a new `src/sddb/cohort.py` as `class SpatialDataCohort(Sequence[Dataset])`, keeping every method identical. Add `from __future__ import annotations` and the imports it needs (`pandas`, `Sequence`/`Iterator`, `Dataset`, `Source`, `Path`). In `dataset.py`, leave `Dataset` and `Source`; import `SpatialDataCohort` lazily inside methods only if needed (avoid a cycle — `cohort.py` imports from `dataset.py`, not vice versa).

- [ ] **Step 2: Mechanical rename across call sites.**

```bash
cd /Users/tim.treis/Documents/GitHub/spatialdata-db-client
grep -rl '\bResults\b' src tests | while read f; do
  sed -i '' 's/\bResults\b/SpatialDataCohort/g' "$f"
done
# fix imports: Results used to come from sddb.dataset; it now lives in sddb.cohort
grep -rn 'from sddb.dataset import' src tests | grep SpatialDataCohort
```
Then by hand: change `from sddb.dataset import SpatialDataCohort` → `from sddb.cohort import SpatialDataCohort` wherever the sed left it pointing at `dataset`; in `catalog.py` the `query`/`search` return construction `Results(...)` is now `SpatialDataCohort(...)`; update `src/sddb/__init__.py` export list and import.

- [ ] **Step 3: Update `__init__.py`.**

```python
from sddb.catalog import Catalog
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset
from sddb.manifest import Manifest, ManifestVersionMismatch
from sddb.remote import open_sdata

__all__ = ["Catalog", "Dataset", "Manifest", "ManifestVersionMismatch", "SpatialDataCohort", "__version__", "open_sdata"]
```

- [ ] **Step 4: Run the full suite — everything must still pass after the rename.**

Run: `.venv/bin/pytest -q`
Expected: PASS (same count as before the rename). Then `.venv/bin/mypy src` and `.venv/bin/ruff check src tests` clean.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: rename Results -> SpatialDataCohort, move to cohort.py"
```

---

## Task 2: Cohort inspection (`summary`/`groupby`/`modalities`/`licenses`/`coverage`)

**Files:**
- Modify: `src/sddb/cohort.py`
- Test: `tests/test_cohort_inspect.py`

**Interfaces:**
- Produces free functions in `cohort.py` and thin methods on `SpatialDataCohort`:
  - `group_counts(df, field) -> pd.DataFrame` (columns `[field, "n"]`, includes an explicit `"unknown"` row for NA)
  - `modalities(df) -> pd.DataFrame` (columns `["modality", "present"]`, from catalog boolean flags only)
  - `licenses(df) -> pd.DataFrame`, `coverage(df) -> pd.DataFrame` (columns `["field","present","unknown"]`), `summary(df) -> dict`
  - methods `SpatialDataCohort.groupby/modalities/licenses/coverage/summary` delegate to these.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cohort_inspect.py
from __future__ import annotations
import pandas as pd
from sddb.cohort import SpatialDataCohort
from tests._fixtures import make_fixture_catalog

def _cohort():
    df = make_fixture_catalog()
    df.loc[df["uid"] == "uid0004", "study_id"] = pd.NA  # one unknown study
    return SpatialDataCohort(df)

def test_groupby_includes_unknown_bucket():
    out = _cohort().groupby("study_id")
    assert "unknown" in set(out["study_id"])
    assert int(out.loc[out["study_id"] == "unknown", "n"].iloc[0]) == 1
    assert out["n"].sum() == 5  # every member counted exactly once

def test_modalities_uses_catalog_flags_no_network(monkeypatch):
    import sddb.remote as remote
    monkeypatch.setattr(remote, "elements", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network!")))
    out = _cohort().modalities()  # must not touch remote.elements
    assert set(out.columns) == {"modality", "present"}

def test_coverage_counts_present_and_unknown():
    cov = _cohort().coverage()
    row = cov.loc[cov["field"] == "study_id"].iloc[0]
    assert int(row["present"]) == 4 and int(row["unknown"]) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_cohort_inspect.py -q`
Expected: FAIL (`AttributeError: 'SpatialDataCohort' object has no attribute 'groupby'`).

- [ ] **Step 3: Implement the free functions + methods in `cohort.py`**

```python
# module-level in cohort.py
_MODALITY_FLAGS = ("hne_image", "if_image", "ftu_annotation")  # catalog boolean columns

def group_counts(df: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in df.columns:
        raise KeyError(f"no such field: {field!r}")
    key = df[field].astype("string").fillna("unknown")
    out = key.value_counts(dropna=False).rename_axis(field).reset_index(name="n")
    return out.sort_values("n", ascending=False, ignore_index=True)

def modalities(df: pd.DataFrame) -> pd.DataFrame:
    rows = [(flag, bool(df[flag].fillna(False).any())) for flag in _MODALITY_FLAGS if flag in df.columns]
    return pd.DataFrame(rows, columns=["modality", "present"])

def licenses(df: pd.DataFrame) -> pd.DataFrame:
    return group_counts(df, "license") if "license" in df.columns else pd.DataFrame(columns=["license", "n"])

def coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        present = int(df[col].notna().sum())
        rows.append((col, present, len(df) - present))
    return pd.DataFrame(rows, columns=["field", "present", "unknown"])

def summary(df: pd.DataFrame) -> dict:
    return {
        "n_objects": len(df),
        "n_studies": int(df["study_id"].astype("string").fillna("unknown").nunique()) if "study_id" in df else None,
        "technologies": sorted(df["technology"].dropna().unique()) if "technology" in df else [],
    }
```
Then on the class:
```python
    def groupby(self, field: str) -> pd.DataFrame: return group_counts(self._df, field)
    def modalities(self) -> pd.DataFrame: return modalities(self._df)
    def licenses(self) -> pd.DataFrame: return licenses(self._df)
    def coverage(self) -> pd.DataFrame: return coverage(self._df)
    def summary(self) -> dict: return summary(self._df)
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest tests/test_cohort_inspect.py -q` → PASS. Then `mypy`/`ruff` clean.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(cohort): inspection (groupby/modalities/licenses/coverage/summary)"`

---

## Task 3 (C0 spike): Determine lamin version pinning + URL mutability

**Files:**
- Create: `docs/superpowers/notes/C0-lamin-pinning.md` (findings; throwaway probe script stays out of `src/`)

**This is a spike — the output is a recorded finding, not kept code.**

- [ ] **Step 1: Probe a real store's root attrs and URL shape**

```bash
.venv/bin/python - <<'PY'
import zarr
from sddb.catalog import Catalog
from sddb.remote import storage_options
c = Catalog()  # network
df = c.to_df()
url = df["zarr_url"].iloc[0]
print("zarr_url:", url)
g = zarr.open_group(url, mode="r", storage_options=storage_options(url) or None)
print("root attrs keys:", list(dict(g.attrs).keys()))
print("root attrs:", dict(g.attrs))
PY
```

- [ ] **Step 2: Record the finding** in `docs/superpowers/notes/C0-lamin-pinning.md`: (a) is `zarr_url` version-qualified (does the path/uid encode a version) or a mutable pointer? (b) which field, if any, carries the lamin version (a root-attrs key, or the uid suffix)? (c) the chosen `lamin_version` extraction rule for Task 4.

- [ ] **Step 3: Commit the note** — `git add docs/superpowers/notes/C0-lamin-pinning.md && git commit -m "docs: C0 lamin version pinning findings"`

> Whatever C0 finds, verify-on-read (Task 6) still ships — a version-qualified URL makes drift unlikely; a mutable one makes verification essential. Task 4 uses the extraction rule recorded here; if none is found, `lamin_version` is `None` and the fingerprint carries the guarantee.

---

## Task 4 (C2): `objectmeta.py` — metadata-only per-object fetch + fingerprint

**Files:**
- Create: `src/sddb/objectmeta.py`
- Test: `tests/test_objectmeta.py`

**Interfaces:**
- Consumes: `sddb.remote.elements(zarr_url) -> dict[str, dict]`, `storage_options`.
- Produces:
  - `@dataclass(frozen=True) ObjectMeta(uid: str, zarr_url: str, lamin_version: str | None, elements: dict[str, dict], tables: tuple[str, ...], fingerprint: str)`
  - `fetch_object_metadata(uid: str, zarr_url: str) -> ObjectMeta`
  - `fetch_many(pairs: list[tuple[str, str]], *, workers: int = 8) -> list[ObjectMeta]` — raises `ObjectMetaError(failed: dict[str,str])` if ANY fetch fails.
  - `compute_fingerprint(elements: dict, tables: tuple[str,...], lamin_version: str | None) -> str`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_objectmeta.py
from __future__ import annotations
import pytest
from sddb.objectmeta import ObjectMeta, fetch_object_metadata, fetch_many, compute_fingerprint, ObjectMetaError
from tests._fixtures import make_tiny_sdata_zarr

def test_fetch_reads_metadata_only(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    m = fetch_object_metadata("u1", str(z))
    assert m.uid == "u1"
    assert "tables/table" in m.elements          # the 4x3 table element
    assert m.tables == ("table",)
    assert m.fingerprint.startswith("sha256:")

def test_fingerprint_changes_with_shape(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    base = fetch_object_metadata("u1", str(z)).fingerprint
    mutated = compute_fingerprint({"tables/table": {"type": "tables", "shape": (5, 3), "dtype": "float32"}}, ("table",), None)
    assert base != mutated

def test_fetch_many_all_or_nothing(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    with pytest.raises(ObjectMetaError) as ei:
        fetch_many([("good", str(z)), ("bad", str(tmp_path / "missing.zarr"))], workers=2)
    assert "bad" in ei.value.failed
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/pytest tests/test_objectmeta.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `objectmeta.py`**

```python
from __future__ import annotations
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from sddb.remote import elements

class ObjectMetaError(RuntimeError):
    """One or more object-metadata fetches failed; `failed` maps uid -> error string."""
    def __init__(self, failed: dict[str, str]) -> None:
        self.failed = failed
        super().__init__(f"metadata fetch failed for {len(failed)} object(s): {sorted(failed)}")

@dataclass(frozen=True)
class ObjectMeta:
    uid: str
    zarr_url: str
    lamin_version: str | None
    elements: dict[str, dict]
    tables: tuple[str, ...]
    fingerprint: str

def compute_fingerprint(elements: dict, tables: tuple[str, ...], lamin_version: str | None) -> str:
    payload = json.dumps(
        {"elements": {k: elements[k] for k in sorted(elements)}, "tables": sorted(tables), "lamin_version": lamin_version},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

def _lamin_version(zarr_url: str) -> str | None:
    # Per C0 (docs/superpowers/notes/C0-lamin-pinning.md): fill in the recorded extraction rule.
    # Fallback when none found: None (fingerprint carries the drift guarantee).
    return None

def fetch_object_metadata(uid: str, zarr_url: str) -> ObjectMeta:
    el = elements(zarr_url)  # raises FileNotFoundError if the store cannot be opened; metadata only
    tables = tuple(sorted(k.split("/", 1)[1] for k in el if k.startswith("tables/")))
    ver = _lamin_version(zarr_url)
    return ObjectMeta(uid, zarr_url, ver, el, tables, compute_fingerprint(el, tables, ver))

def fetch_many(pairs: list[tuple[str, str]], *, workers: int = 8) -> list[ObjectMeta]:
    out: dict[str, ObjectMeta] = {}
    failed: dict[str, str] = {}
    def one(p: tuple[str, str]) -> None:
        uid, url = p
        try:
            out[uid] = fetch_object_metadata(uid, url)
        except Exception as e:  # noqa: BLE001 — aggregated and re-raised as ObjectMetaError
            failed[uid] = f"{type(e).__name__}: {e}"
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, pairs))
    if failed:
        raise ObjectMetaError(failed)
    return [out[uid] for uid, _ in pairs]
```

- [ ] **Step 4: Run to verify they pass** — `.venv/bin/pytest tests/test_objectmeta.py -q` → PASS. `mypy`/`ruff` clean.

- [ ] **Step 5: Commit** — `git commit -am "feat(objectmeta): metadata-only fetch + fingerprint (all-or-nothing)"`

---

## Task 5 (C3a): `FrozenCohort` + hash + `write`/`load` + `SpatialDataCohort.freeze()`

**Files:**
- Create: `src/sddb/freeze.py`
- Modify: `src/sddb/cohort.py` (add `object_metadata()` + `freeze()`)
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `objectmeta.fetch_many`, `SpatialDataCohort._df`/`_source`/`_filter`, `dataset.Source`.
- Produces:
  - `cohort_hash(filter: dict, members: pd.DataFrame) -> str` — sha256 over `{filter, members[uid,lamin_version,fingerprint] sorted by uid}`, excludes timestamps.
  - `@dataclass(frozen=True) FrozenCohort(schema_version, created_at, source, filter, hash, members)`; `.write(path)`, `FrozenCohort.load(path)`.
  - `SpatialDataCohort._filter: dict` (set by `catalog.query/search`), `SpatialDataCohort.object_metadata(workers=8) -> pd.DataFrame`, `SpatialDataCohort.freeze() -> FrozenCohort`.

- [ ] **Step 1: Record the filter on the cohort.** In `catalog.py`, pass the query kwargs/search text to the constructor: `SpatialDataCohort(df[keep], matched=matched, source=..., filter={...})`. Add `filter: dict | None = None` to `SpatialDataCohort.__init__`, store `self._filter = filter or {}`. (Update Task 1's constructor interface note accordingly.)

- [ ] **Step 2: Write failing tests**

```python
# tests/test_freeze.py
from __future__ import annotations
import pandas as pd
from sddb.freeze import FrozenCohort, cohort_hash

def _members():
    return pd.DataFrame({
        "uid": ["u2", "u1"], "lamin_version": [None, None],
        "fingerprint": ["sha256:b", "sha256:a"], "study_id": ["S2", "S1"], "zarr_url": ["z2", "z1"],
    })

def test_hash_is_order_independent_and_excludes_timestamp():
    m = _members()
    h1 = cohort_hash({"organism": "human"}, m)
    h2 = cohort_hash({"organism": "human"}, m.iloc[::-1].reset_index(drop=True))
    assert h1 == h2 and h1.startswith("sha256:")

def test_hash_changes_with_membership():
    m = _members()
    assert cohort_hash({}, m) != cohort_hash({}, m.iloc[:1])

def test_write_load_roundtrip(tmp_path):
    from sddb.dataset import Source
    fc = FrozenCohort("1", "2026-10-07T00:00:00Z", Source(url="cat", version="v"), {"organism": "human"},
                      cohort_hash({"organism": "human"}, _members()), _members())
    p = fc.write(tmp_path / "cohort.json")
    back = FrozenCohort.load(p)
    assert back.hash == fc.hash and list(back.members["uid"]) == list(fc.members["uid"])
    assert back.filter == {"organism": "human"}
```

- [ ] **Step 3: Run to verify they fail** — `.venv/bin/pytest tests/test_freeze.py -q` → FAIL (module missing).

- [ ] **Step 4: Implement `freeze.py`**

```python
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd
from sddb.dataset import Source

SCHEMA_VERSION = "1"
_SIDECAR_THRESHOLD = 200  # members beyond this go to a parquet sidecar

def cohort_hash(filter: dict, members: pd.DataFrame) -> str:
    rows = [
        {"uid": str(r.uid), "lamin_version": (None if pd.isna(r.lamin_version) else str(r.lamin_version)),
         "fingerprint": str(r.fingerprint)}
        for r in members.sort_values("uid").itertuples(index=False)
    ]
    payload = json.dumps({"filter": filter, "members": rows}, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

@dataclass(frozen=True)
class FrozenCohort:
    schema_version: str
    created_at: str
    source: Source
    filter: dict
    hash: str
    members: pd.DataFrame  # read-only by contract; frozen=True does NOT protect in-place mutation

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        head = {"schema_version": self.schema_version, "created_at": self.created_at,
                "source": {"url": self.source.url, "version": self.source.version},
                "filter": self.filter, "hash": self.hash, "n_members": len(self.members)}
        if len(self.members) > _SIDECAR_THRESHOLD:
            side = p.with_suffix(".members.parquet")
            self.members.to_parquet(side)
            head["members_ref"] = side.name
        else:
            head["members"] = json.loads(self.members.to_json(orient="records"))
        p.write_text(json.dumps(head, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> FrozenCohort:
        p = Path(path)
        head = json.loads(p.read_text())
        if "members_ref" in head:
            members = pd.read_parquet(p.parent / head["members_ref"])
        else:
            members = pd.DataFrame(head["members"])
        src = Source(url=head["source"]["url"], version=head["source"]["version"])
        return cls(head["schema_version"], head["created_at"], src, head["filter"], head["hash"], members)
```

- [ ] **Step 5: Run to verify they pass** — `.venv/bin/pytest tests/test_freeze.py -q` → PASS.

- [ ] **Step 6: Wire `freeze()` on the cohort** in `cohort.py`:

```python
    def object_metadata(self, *, workers: int = 8) -> pd.DataFrame:
        from sddb.objectmeta import fetch_many
        pairs = [(str(r.uid), str(r.zarr_url)) for r in self._df.itertuples(index=False)]
        metas = fetch_many(pairs, workers=workers)
        return pd.DataFrame([{"uid": m.uid, "lamin_version": m.lamin_version, "fingerprint": m.fingerprint,
                              "tables": list(m.tables)} for m in metas])

    def freeze(self, *, workers: int = 8) -> "FrozenCohort":
        from datetime import UTC, datetime
        from sddb.freeze import SCHEMA_VERSION, FrozenCohort, cohort_hash
        meta = self.object_metadata(workers=workers)
        members = self._df.merge(meta, on="uid", how="left")
        return FrozenCohort(SCHEMA_VERSION, datetime.now(UTC).isoformat(), self._source, dict(self._filter),
                            cohort_hash(dict(self._filter), members), members)
```
Add a freeze integration test against a local fixture store (build a 1-row cohort whose `zarr_url` points at `make_tiny_sdata_zarr`, freeze, assert `members` has a `fingerprint`).

- [ ] **Step 7: Run suite + mypy/ruff; Commit** — `git commit -am "feat(freeze): FrozenCohort + deterministic hash + write/load + cohort.freeze()"`

---

## Task 6 (C3b): Verify-on-read — `CohortDriftError`

**Files:**
- Modify: `src/sddb/freeze.py` (add `verify_members`, `CohortDriftError`)
- Test: `tests/test_freeze_verify.py`

**Interfaces:**
- Produces: `class CohortDriftError(RuntimeError)`; `verify_members(members: pd.DataFrame, *, workers: int = 8) -> None` — re-fetches each member's metadata, recomputes fingerprint, raises `CohortDriftError(drifted: dict[str,str])` on any mismatch. Consumed by `to_anndata`/`build` (Tasks 7 & 9) via a `verify: bool = True` flag.

- [ ] **Step 1: Write failing test** (freeze a fixture store, then mutate its table shape on disk → verify raises)

```python
# tests/test_freeze_verify.py
from __future__ import annotations
import anndata as ad, numpy as np, pandas as pd, pytest
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from sddb.cohort import SpatialDataCohort
from sddb.freeze import verify_members, CohortDriftError

def _write(path, n_obs):
    img = Image2DModel.parse(np.zeros((3, 8, 8), "uint8"), dims=("c", "y", "x"))
    SpatialData(images={"img": img}, tables={"table": ad.AnnData(X=np.ones((n_obs, 3), "float32"))}).write(path)
    return path

def test_verify_passes_then_detects_drift(tmp_path):
    z = _write(tmp_path / "o.zarr", 4)
    df = pd.DataFrame({"uid": ["o"], "zarr_url": [str(z)], "study_id": ["S"]})
    fc = SpatialDataCohort(df, filter={}).freeze()
    verify_members(fc.members)  # no raise: unchanged
    import shutil; shutil.rmtree(z); _write(z, 9)  # republish different data at the same path
    with pytest.raises(CohortDriftError) as ei:
        verify_members(fc.members)
    assert "o" in ei.value.drifted
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/pytest tests/test_freeze_verify.py -q` → FAIL (`verify_members` undefined).

- [ ] **Step 3: Implement**

```python
class CohortDriftError(RuntimeError):
    """A frozen cohort's objects changed since freeze; `drifted` maps uid -> 'reason'."""
    def __init__(self, drifted: dict[str, str]) -> None:
        self.drifted = drifted
        super().__init__(f"{len(drifted)} object(s) drifted since freeze: {sorted(drifted)}")

def verify_members(members: pd.DataFrame, *, workers: int = 8) -> None:
    from sddb.objectmeta import fetch_many
    pairs = [(str(r.uid), str(r.zarr_url)) for r in members.itertuples(index=False)]
    now = {m.uid: m for m in fetch_many(pairs, workers=workers)}
    drifted: dict[str, str] = {}
    for r in members.itertuples(index=False):
        cur = now[str(r.uid)]
        if cur.fingerprint != str(r.fingerprint):
            drifted[str(r.uid)] = f"fingerprint {r.fingerprint} -> {cur.fingerprint}"
    if drifted:
        raise CohortDriftError(drifted)
```

- [ ] **Step 4: Run to verify it passes** — `.venv/bin/pytest tests/test_freeze_verify.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(freeze): verify-on-read drift detection (CohortDriftError)"`

---

## Task 7 (C4): Lazy table concat — `to_anndata()`

**Files:**
- Create: `src/sddb/concat.py`
- Modify: `src/sddb/cohort.py` + `src/sddb/freeze.py` (add `to_anndata`)
- Test: `tests/test_concat.py`

**Interfaces:**
- Consumes: `remote.storage_options`, `freeze.verify_members`.
- Produces: `concat_tables(members: pd.DataFrame, *, table: str | None = None, verify: bool = True, storage=...) -> AnnData` — reads only each store's `tables/<name>` group (default per-member `default_table` if present else `"table"`), `anndata.concat(join="outer", label="uid", keys=uids)`. Raises `KeyError` naming a member whose requested table is absent. `SpatialDataCohort.to_anndata(table=None)` and `FrozenCohort.to_anndata(table=None, verify=True)` delegate.

- [ ] **Step 1: Write failing test**

```python
# tests/test_concat.py
from __future__ import annotations
import anndata as ad, numpy as np, pandas as pd, pytest
from spatialdata import SpatialData
from sddb.concat import concat_tables

def _store(path, n):
    SpatialData(tables={"table": ad.AnnData(X=np.ones((n, 3), "float32"))}).write(path)
    return str(path)

def test_concat_reads_only_tables(tmp_path):
    a = _store(tmp_path / "a.zarr", 4); b = _store(tmp_path / "b.zarr", 6)
    members = pd.DataFrame({"uid": ["a", "b"], "zarr_url": [a, b],
                            "fingerprint": ["x", "y"], "default_table": ["table", "table"]})
    adata = concat_tables(members, verify=False)
    assert adata.n_obs == 10 and set(adata.obs["uid"]) == {"a", "b"}

def test_concat_missing_table_names_member(tmp_path):
    a = _store(tmp_path / "a.zarr", 4)
    members = pd.DataFrame({"uid": ["a"], "zarr_url": [a], "fingerprint": ["x"], "default_table": ["nope"]})
    with pytest.raises(KeyError) as ei:
        concat_tables(members, verify=False)
    assert "a" in str(ei.value)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (module missing).

- [ ] **Step 3: Implement `concat.py`** (open each `tables/<name>` via `anndata.read_zarr` on the sub-path; zarr metadata + table arrays only, never the images):

```python
from __future__ import annotations
import anndata as ad
import pandas as pd
from sddb.remote import storage_options

def _table_name(row) -> str:
    dt = getattr(row, "default_table", None)
    return str(dt) if isinstance(dt, str) and dt else "table"

def concat_tables(members: pd.DataFrame, *, table: str | None = None, verify: bool = True) -> ad.AnnData:
    if verify:
        from sddb.freeze import verify_members
        verify_members(members)
    parts, keys = [], []
    for row in members.itertuples(index=False):
        name = table or _table_name(row)
        url = f"{str(row.zarr_url).rstrip('/')}/tables/{name}"
        try:
            parts.append(ad.read_zarr(url))  # storage_options applied by anndata/zarr via fsspec url
        except Exception as e:  # noqa: BLE001
            raise KeyError(f"member {row.uid!r}: table {name!r} not readable at {url}: {e}") from e
        keys.append(str(row.uid))
    return ad.concat(parts, join="outer", label="uid", keys=keys, index_unique="-")
```
> Note for the implementer: if `ad.read_zarr` on a remote sub-path needs explicit `storage_options`, pass them via `zarr.open_group(url, storage_options=storage_options(url))` and `ad.read_zarr(group)`. Local-path tests exercise the happy path; a `@pytest.mark.network` test covers remote.

- [ ] **Step 4: Run to verify it passes** — PASS. Add delegators: `SpatialDataCohort.to_anndata(self, *, table=None)` → `concat_tables(self._df, table=table, verify=False)` (live cohort isn't frozen, nothing to verify against); `FrozenCohort.to_anndata(self, *, table=None, verify=True)` → `concat_tables(self.members, table=table, verify=verify)`.

- [ ] **Step 5: Commit** — `git commit -am "feat(concat): lazy per-cohort table concat -> AnnData"`

---

## Task 8 (C5): `SplitManifest` + deterministic grouped split

**Files:**
- Create: `src/sddb/split.py`
- Modify: `src/sddb/freeze.py` (add `FrozenCohort.split`)
- Test: `tests/test_split.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) SplitManifest(parent_hash, by, seed, ratios, mode, assignments: dict[str,str])`; `.write(path)`, `.load(path)`.
  - `make_split(members, parent_hash, *, by="study_id", train, val, test, seed, mode="group") -> SplitManifest`.
  - `FrozenCohort.split(**kw) -> SplitManifest` (passes `self.members`, `self.hash`).

- [ ] **Step 1: Write failing tests** (determinism, no leakage, small-cohort raise, annotate mode)

```python
# tests/test_split.py
from __future__ import annotations
import pandas as pd, pytest
from sddb.split import make_split

def _members(studies):
    return pd.DataFrame({"uid": [f"u{i}" for i in range(len(studies))], "study_id": studies})

def test_deterministic_and_group_safe():
    m = _members(["A","A","B","C","D","E","F","G","H","I"])
    s1 = make_split(m, "h", train=0.8, val=0.1, test=0.1, seed=42)
    s2 = make_split(m, "h", train=0.8, val=0.1, test=0.1, seed=42)
    assert s1.assignments == s2.assignments
    # no study spans two folds
    fold_of = s1.assignments
    by_study = {}
    for u, st in zip(m["uid"], m["study_id"]):
        by_study.setdefault(st, set()).add(fold_of[u])
    assert all(len(f) == 1 for f in by_study.values())

def test_missing_study_is_singleton():
    m = pd.DataFrame({"uid": ["u0","u1"], "study_id": [pd.NA, "A"]})
    s = make_split(m, "h", train=0.5, val=0.0, test=0.5, seed=1)
    assert set(s.assignments) == {"u0", "u1"}

def test_too_few_groups_raises():
    m = _members(["A", "A"])  # 1 group, 3 non-zero folds
    with pytest.raises(ValueError):
        make_split(m, "h", train=0.34, val=0.33, test=0.33, seed=1)

def test_annotate_mode_labels_only():
    m = _members(["A","B"])
    s = make_split(m, "h", train=0.5, val=0.0, test=0.5, seed=1, mode="annotate")
    assert s.assignments == {"u0": "A", "u1": "B"}
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module missing).

- [ ] **Step 3: Implement `split.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SplitManifest:
    parent_hash: str
    by: str
    seed: int
    ratios: dict
    mode: str
    assignments: dict[str, str]

    def write(self, path: str | Path) -> Path:
        p = Path(path); p.write_text(json.dumps(self.__dict__, indent=2)); return p

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        d = json.loads(Path(path).read_text()); return cls(**d)

def _group_key(uid, study) -> str:
    return f"object:{uid}" if (study is None or pd.isna(study)) else str(study)

def make_split(members: pd.DataFrame, parent_hash: str, *, by: str = "study_id",
               train: float, val: float, test: float, seed: int, mode: str = "group") -> SplitManifest:
    ratios = {"train": train, "val": val, "test": test}
    groups = {str(r.uid): _group_key(r.uid, getattr(r, by, None)) for r in members.itertuples(index=False)}
    if mode == "annotate":
        return SplitManifest(parent_hash, by, seed, ratios, mode, groups)
    keys = sorted(set(groups.values()))                       # deterministic input order
    nonzero = [f for f, w in ratios.items() if w > 0]
    if len(keys) < len(nonzero):
        raise ValueError(f"cohort has {len(keys)} group(s) but {len(nonzero)} non-zero folds requested "
                         f"({nonzero}); too few to fill every fold without an empty one")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(keys))
    counts = _largest_remainder(len(keys), ratios)            # guarantees >=1 per non-zero fold
    fold_by_key: dict[str, str] = {}
    i = 0
    for fold in ("train", "val", "test"):
        for _ in range(counts[fold]):
            fold_by_key[order[i]] = fold; i += 1
    return SplitManifest(parent_hash, by, seed, ratios, mode, {u: fold_by_key[g] for u, g in groups.items()})

def _largest_remainder(n: int, ratios: dict) -> dict:
    raw = {f: ratios[f] * n for f in ("train", "val", "test")}
    base = {f: int(raw[f]) for f in raw}
    for f, w in ratios.items():
        if w > 0 and base[f] == 0:
            base[f] = 1
    while sum(base.values()) > n:                              # trim overflow from the largest fold
        base[max(base, key=lambda k: base[k])] -= 1
    while sum(base.values()) < n:                              # give remainder to the largest ratio
        base[max(ratios, key=lambda k: ratios[k])] += 1
    return base
```

- [ ] **Step 4: Run to verify they pass** — PASS. Add `FrozenCohort.split(self, **kw)` → `make_split(self.members, self.hash, **kw)`; add a test that a loaded split's `parent_hash` equals the cohort's hash and that applying a split whose `parent_hash` mismatches raises (guard in a thin `apply` helper or document the check at use site).

- [ ] **Step 5: Commit** — `git commit -am "feat(split): deterministic group-safe SplitManifest + small-cohort guard"`

---

## Task 9 (C6+C7): `TaskAdapter` protocol, two-tier `check()`, `build(materialize=)` + provenance

**Files:**
- Create: `src/sddb/adapter.py`
- Modify: `src/sddb/freeze.py` (add `FrozenCohort.check`/`build`)
- Test: `tests/test_adapter.py`

**Interfaces:**
- Produces:
  - `Issue = str`; `@dataclass PreflightReport(total, compatible, excluded: dict[str,list[str]], tier: str)` with `__repr__`.
  - `class TaskAdapter(Protocol)`: `requirements()`, `validate_meta(meta: ObjectMeta) -> list[Issue]`, `validate(sdata) -> list[Issue]`, `build(sdata) -> Any`.
  - `check(members, adapter, *, deep=False) -> PreflightReport`.
  - `class TaskView` (iterable of `(uid, thunk)`); `build(members, adapter, *, split=None, materialize=False, out=None, verify=True) -> TaskView`; writes `out/provenance.json` on materialize.
  - `FrozenCohort.check`/`FrozenCohort.build` delegate (passing `self.members`, `self.hash`).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adapter.py
from __future__ import annotations
import json, pandas as pd
from sddb.objectmeta import ObjectMeta
from sddb.adapter import check, build, PreflightReport

class FakeAdapter:
    name = "fake"; version = "0.1"; config = {"k": 1}
    def requirements(self): return {"tables": ["table"]}
    def validate_meta(self, meta): return [] if "tables/table" in meta.elements else ["missing table"]
    def validate(self, sdata): return []
    def build(self, sdata): return {"n": 1}

def _meta(uid, has_table):
    el = {"tables/table": {}} if has_table else {"images/img": {}}
    return ObjectMeta(uid, f"z/{uid}", None, el, ("table",) if has_table else (), "sha256:x")

def test_check_advisory_tier_counts_and_reasons(monkeypatch):
    members = pd.DataFrame({"uid": ["a", "b"], "zarr_url": ["za", "zb"], "fingerprint": ["x", "y"]})
    monkeypatch.setattr("sddb.adapter._metas",
                        lambda m, **k: [_meta("a", True), _meta("b", False)])
    rep = check(members, FakeAdapter(), deep=False)
    assert rep.total == 2 and rep.compatible == 1 and rep.tier == "advisory"
    assert "b" in rep.excluded["missing table"]

def test_build_materialize_writes_provenance(tmp_path, monkeypatch):
    members = pd.DataFrame({"uid": ["a"], "zarr_url": ["za"], "fingerprint": ["x"], "default_table": ["table"]})
    monkeypatch.setattr("sddb.adapter._open", lambda url: object())  # skip real sdata open
    build(members, FakeAdapter(), materialize=True, out=tmp_path, verify=False, cohort_hash="h", split_hash=None)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["cohort_hash"] == "h" and prov["adapter"]["name"] == "fake" and prov["adapter"]["version"] == "0.1"
```

- [ ] **Step 2: Run to verify they fail** — FAIL (module missing).

- [ ] **Step 3: Implement `adapter.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import pandas as pd
import sddb
from sddb.objectmeta import ObjectMeta, fetch_many

Issue = str

@runtime_checkable
class TaskAdapter(Protocol):
    name: str
    version: str
    config: dict
    def requirements(self) -> dict: ...
    def validate_meta(self, meta: ObjectMeta) -> list[Issue]: ...
    def validate(self, sdata: Any) -> list[Issue]: ...
    def build(self, sdata: Any) -> Any: ...

@dataclass
class PreflightReport:
    total: int
    compatible: int
    excluded: dict[str, list[Issue]]   # reason -> uids
    tier: str                          # "advisory" | "authoritative"
    def __repr__(self) -> str:
        lines = [f"{self.total} candidate / {self.compatible} compatible  [{self.tier}]"]
        lines += [f"  {len(uids)} {reason}: {', '.join(uids)}" for reason, uids in sorted(self.excluded.items())]
        return "\n".join(lines)

def _metas(members: pd.DataFrame, **kw) -> list[ObjectMeta]:
    return fetch_many([(str(r.uid), str(r.zarr_url)) for r in members.itertuples(index=False)], **kw)

def _open(url: str) -> Any:
    from sddb.remote import open_sdata
    return open_sdata(url, lazy=False)

def check(members: pd.DataFrame, adapter: TaskAdapter, *, deep: bool = False) -> PreflightReport:
    excluded: dict[str, list[Issue]] = {}
    compatible = 0
    if deep:
        for r in members.itertuples(index=False):
            issues = adapter.validate(_open(str(r.zarr_url)))
            if issues:
                for i in issues: excluded.setdefault(i, []).append(str(r.uid))
            else:
                compatible += 1
        return PreflightReport(len(members), compatible, excluded, "authoritative")
    for meta in _metas(members):
        issues = adapter.validate_meta(meta)
        if issues:
            for i in issues: excluded.setdefault(i, []).append(meta.uid)
        else:
            compatible += 1
    return PreflightReport(len(members), compatible, excluded, "advisory")

class TaskView:
    def __init__(self, items: list[tuple[str, Any]]) -> None: self._items = items
    def __iter__(self): return iter(self._items)
    def __len__(self): return len(self._items)
    def __getitem__(self, i): return self._items[i]

def build(members: pd.DataFrame, adapter: TaskAdapter, *, split=None, materialize: bool = False,
          out: str | Path | None = None, verify: bool = True, cohort_hash: str = "", split_hash=None) -> TaskView:
    if verify:
        from sddb.freeze import verify_members
        verify_members(members)
    def thunk(url: str): return lambda: adapter.build(_open(url))
    view = TaskView([(str(r.uid), thunk(str(r.zarr_url))) for r in members.itertuples(index=False)])
    if not materialize:
        return view
    out_dir = Path(out or "."); out_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for uid, fn in view:
        result = fn()
        produced.append({"uid": uid, "output": repr(type(result).__name__)})
    (out_dir / "provenance.json").write_text(json.dumps({
        "cohort_hash": cohort_hash, "split_hash": split_hash,
        "adapter": {"name": adapter.name, "version": adapter.version, "config": adapter.config},
        "sddb_version": sddb.__version__, "members": produced,
    }, indent=2))
    return view
```
- [ ] **Step 4: Run to verify they pass** — `.venv/bin/pytest tests/test_adapter.py -q` → PASS.

- [ ] **Step 5: Wire on `FrozenCohort`** in `freeze.py`:

```python
    def check(self, adapter, *, deep: bool = False):
        from sddb.adapter import check
        return check(self.members, adapter, deep=deep)

    def build(self, adapter, *, split=None, materialize: bool = False, out=None, verify: bool = True):
        from sddb.adapter import build
        return build(self.members, adapter, split=split, materialize=materialize, out=out, verify=verify,
                     cohort_hash=self.hash, split_hash=(split.parent_hash if split else None))
```

- [ ] **Step 6: Run full suite + mypy/ruff; Commit** — `git commit -am "feat(adapter): TaskAdapter protocol, two-tier check, build+provenance"`

---

## Task 10: Docs + changelog

**Files:**
- Modify: `docs/` (a `query → freeze → build` tutorial page), `README` if it lists the API surface.
- Test: `sphinx-build` clean.

- [ ] **Step 1:** Add a short myst/rst page walking `Catalog().query(...) → freeze() → split() → check(adapter) → build(adapter, materialize=True)`, noting verify-on-read and that `Results` is now `SpatialDataCohort`.
- [ ] **Step 2:** Run `.venv/bin/python -m sphinx -W -b html docs docs/_build/html` → clean.
- [ ] **Step 3: Commit** — `git commit -am "docs: cohort layer tutorial (query -> freeze -> build)"`

---

## Self-review notes (for the executor)

- **Spec coverage:** C1→Tasks 1–2; C0→Task 3; C2→Task 4; C3→Tasks 5–6 (freeze + verify-on-read); C4→Task 7; C5→Task 8; C6/C7→Task 9; §11 testing distributed into each task; docs→Task 10. Out-of-scope items (CIVET, reference layer, engine E1–E3) are intentionally absent.
- **Type consistency:** `SpatialDataCohort(df, *, matched, source, filter)`, `ObjectMeta(uid, zarr_url, lamin_version, elements, tables, fingerprint)`, `cohort_hash(filter, members)`, `FrozenCohort(schema_version, created_at, source, filter, hash, members)`, `SplitManifest(parent_hash, by, seed, ratios, mode, assignments)`, `PreflightReport(total, compatible, excluded, tier)` — used consistently across tasks.
- **Review-focus tests:** drift (Task 6), freeze all-or-nothing (Task 4), small-cohort split raise + determinism (Task 8), modalities no-network (Task 2). All pinned.
- **Known implementer traps flagged inline:** the `_lamin_version` rule depends on Task 3's finding; `ad.read_zarr` remote `storage_options` (Task 7).
