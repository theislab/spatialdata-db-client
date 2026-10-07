# Cohort layer — `SpatialDataCohort` → freeze → split → build (design)

> Self-contained: a fresh agent can execute this WITHOUT the originating conversation.
> Repo: `theislab/spatialdata-db-client` (public read SDK, `import sddb`, CLI `sddb-cli`, dist
> `spatialdata-db`). Local checkout `/Users/tim.treis/Documents/GitHub/spatialdata-db-client`.
> Strategy source: `~/Downloads/SPATIALDATA_DB_NATURE_METHODS_STRATEGY.md` (Nature Methods Resource
> blueprint). This spec covers the **client-side WP1–4**: cohort semantics, frozen manifests,
> reproducible splitting, and the adapter/build interface. Reference utilities (strategy WP6) and the
> CIVET adapter are explicitly **out of scope** (separate specs).

## 1. Goal

Turn catalog queries into a reproducible scientific workflow:

```
Catalog.query() → SpatialDataCohort → freeze() → FrozenCohort (file+hash)
                                                     ├─ split() → SplitManifest
                                                     ├─ check(adapter) → PreflightReport
                                                     └─ build(adapter, …) → TaskView / materialized artifact
```

The client owns **selection, identity, reproducibility, orchestration**. Adapters (in method repos) own
**task representation**. Method repos own **training**. The stop condition for the abstraction: *adding
another compatible public experiment changes the cohort, not the software.*

## 2. Context — what exists today (do not rebuild)

Read before implementing; the new work sits on top of these.

- `src/sddb/catalog.py` — `Catalog.query(...)` / `Catalog.search(text)` return `Results`. Holds the
  catalog `DataFrame` + a `Source` (url, version). `Catalog.genes` → `GeneIndex`.
- `src/sddb/dataset.py` — `Results(Sequence[Dataset])`: thin `DataFrame` wrapper with `to_df`,
  `citations`/`citations_text`, `download`, slicing/iteration. `Dataset` wraps one row; `.load()`,
  `.elements()`, `.viewer_url`. `Source` dataclass (url, version).
- `src/sddb/manifest.py` — **download** layer: `download(results, dest, …) → Manifest`, resumable,
  `Manifest.refetch`, `ManifestEntry` (uid, zarr_url, dest, size, sha256, status). `_digest` hashes a
  LOCAL store's file paths+sizes. This is a *download record*, NOT the scientific cohort artifact.
- `src/sddb/remote.py` — `open_sdata(zarr_url, lazy=…)`, `storage_options(url)` (anon S3). Remote
  lazy/partial full-SpatialData read is **broken upstream** (spatialdata 0.8 + zarr 3.4; a strict-xfail
  canary flips when fixed). Remote path today = query + `elements()` + `load(lazy=False)` (download then open).
- `src/sddb/citations.py` — `write_citations(results, …)`, filters a published `citations.bib` to the
  `study_id`s present. Consumes a `Results`.
- `src/sddb/_catalog_schema.py` — `CATALOG_COLUMNS`, `validate(df, kind=…)`. Identity columns present:
  `uid` (object, 608/608 unique), `study_id` (536/608, 243 studies), `sample_id` (≈GSM, 571 unique),
  `sddb_id`, `canonical_key`, `collection_name` (71/608, 20 collections), `default_table`, `n_obs`,
  `n_features`, modality flags (`hne_image`, `if_image`, …), `license*`, `zarr_url`.
- `src/sddb/mcp/tools.py`, `src/sddb/cli.py` — consume `Results` (rename target).

### 2.1 Identity model, grounded in the real catalog (608 objects)

| Strategy hierarchy | Our column | Reality |
|---|---|---|
| study | `study_id` | 243 studies; **72 objects (12%) have none**; 59 multi-object studies (max 32). The real leakage unit. |
| donor / source | — | **Absent**, usually unrecoverable from GEO. |
| specimen / section | `sample_id` (≈GSM) | 571 unique for 608 → ~1:1 with the object; weak as a split unit. |
| SpatialData object | `uid` | 608 unique; this is `dataset_uid`. |

The 4-level hierarchy collapses to ~2.5 real levels: `study_id → (sample_id ≈ object) → uid`. Study is
surfaced **metadata**, not an enforced rule. Donor/specimen enrichment is a cross-repo engine ask
(§9), non-blocking.

## 3. Decisions locked in brainstorming

1. **Approach:** thin return object + focused delegating modules (the existing `citations.py`/`manifest.py`
   idiom). **No back-compat alias** — pre-release, `Results` is renamed outright.
2. **Two types:** `SpatialDataCohort` (live, catalog-bound) and `FrozenCohort` (immutable, self-contained).
   The type boundary *is* the reproducibility guarantee.
3. **Missing metadata** stays as explicit `"unknown"`, surfaced in `coverage()`; never silently dropped.
4. **Object identity pinned via lamin version** (hard append-only policy: versions never deleted, only
   superseded). But that policy is **external to the client and unenforceable by it** — the repo already
   notes "a store republished at the same URL is not detected." So the pin alone is not a guarantee:
   **every data read through a `FrozenCohort` re-fetches each object's metadata and verifies
   `lamin_version` + `fingerprint` against the manifest, raising on mismatch** (opt-out only via explicit
   `verify=False`). The `fingerprint` is therefore an *enforced* drift detector on reload, not just a
   recorded canary — though still not a byte-level guarantee (it covers metadata + shapes, not chunk
   contents). **Precondition spike (C0):** determine whether `zarr_url` is version-qualified/immutable or
   a mutable pointer, and identify the version field; the verify-on-read requirement stands either way.
5. **Split surfaces `study_id`** so consumers can hold out / stratify; default groups whole studies into
   one fold (no forced leakage), with an annotate-only mode. Missing `study_id` → `object:<uid>` singleton.
6. **Freeze fetches per-object zarr metadata** (metadata-only, no arrays) — load-bearing for freeze,
   `modalities()`, and tier-1 preflight.
7. **Lazy table concat**: read only each object's `tables/` group → one AnnData; `default_table` default,
   name-overridable, recorded per member. Prototype = AnnData only; multi-table joins deferred.
8. **Frozen cohort = one local file + hash**; promotable to a named DB *collection* later (collection = a
   special case of a frozen cohort). DB *releases* = separate LTS whole-DB snapshots (not this spec).
9. **Adapters:** `TaskAdapter` **Protocol in core**; concrete adapters live in method repos.
10. **Preflight two-tier:** cheap metadata check + optional `deep=True` open-each-sdata.
11. **`build(adapter, materialize=False)` is one method**: lazy plan by default; `materialize=True` writes
    the artifact. Output format owned by the adapter; sddb stamps a uniform `provenance.json`.

## 4. Module layout

```
src/sddb/
  catalog.py     query()/search() → SpatialDataCohort   (change return type)
  cohort.py      SpatialDataCohort (live) + shared inspection free-functions
  frozen.py      FrozenCohort: write/load/hash + split()/check()/build() delegators
  objectmeta.py  per-object zarr metadata fetch (metadata-only)              [C2]
  concat.py      lazy table concat → AnnData                                 [C4]
  split.py       SplitManifest                                               [C5]
  adapter.py     TaskAdapter Protocol, PreflightReport, TaskView, provenance [C6/C7]
  dataset.py     Dataset (unchanged); Results DELETED
  manifest.py    download record (unchanged; distinct from FrozenCohort)
```

Rename `Results` → `SpatialDataCohort` in `catalog.py`, `dataset.py`, `citations.py`, `manifest.py`,
`mcp/tools.py`, `cli.py`, and all tests. Shared pandas inspection (`summary`/`groupby`/`coverage`/…) lives
as free functions in `cohort.py`; both types call them (so `FrozenCohort` inspects without re-querying).

**Two manifests must not be confused:** `manifest.py::Manifest` = bytes-on-disk download record;
`FrozenCohort` = selection + metadata + pinned versions. A `FrozenCohort` MAY reference a download
`Manifest`, but is not one.

## 5. API surface

### 5.1 `SpatialDataCohort` (live, catalog-bound)

```python
class SpatialDataCohort(Sequence[Dataset]):
    # state: _df (members), _filter (exact query kwargs / search text), _source (url+version), matched
    # Sequence: __len__, __iter__, __getitem__ (int → Dataset, slice → SpatialDataCohort)  [as Results today]

    # inspection (pure pandas over _df; "unknown" bucket excluded from % denominators)
    def summary(self) -> pd.DataFrame            # counts by technology/tissue/organism/validation + n studies/objects
    def groupby(self, field: str) -> pd.DataFrame # size per value of `field` incl. an explicit "unknown" row
    def modalities(self) -> pd.DataFrame          # presence per modality flag FROM CATALOG (no network); extensible for multi-omics
    def licenses(self) -> pd.DataFrame            # license id + noncommercial/redistributable/unknown counts
    def coverage(self) -> pd.DataFrame            # per-field completeness: present / unknown / n/a

    # kept from Results
    def to_df(self) -> pd.DataFrame
    def citations(self, path, *, bib_url=None) -> Path
    def citations_text(self, *, bib_url=None) -> tuple[str, int]
    def download(self, dest, *, workers=4, pin_versions=False, allow_version_change=False) -> Manifest

    # new
    def object_metadata(self, *, workers: int = 8) -> pd.DataFrame   # [C2] per-object zarr metadata, cached
    def to_anndata(self, *, table: str | None = None) -> AnnData     # [C4] lazy table concat
    def freeze(self) -> FrozenCohort                                  # [C3] fetch metadata, resolve, hash
```

### 5.2 `FrozenCohort` (immutable, self-contained)

```python
@dataclass(frozen=True)
class FrozenCohort:
    schema_version: str
    created_at: str
    source: Source                 # catalog url + pinned version
    filter: dict                   # the exact query/search that produced it
    hash: str                      # deterministic id (see §7)
    members: pd.DataFrame          # one row per uid: catalog fields + lamin_version + fingerprint + object-meta summary

    @classmethod
    def load(cls, path: str | Path) -> FrozenCohort            # loads the MANIFEST offline; no catalog
    def write(self, path: str | Path) -> Path                   # JSON (+ parquet sidecar for members if large)

    # shared inspection (same free-functions as the live cohort)
    def to_df(self) -> pd.DataFrame
    def summary(self) / groupby(...) / modalities() / coverage() / licenses()
    def to_anndata(self, *, table=None, verify=True) -> AnnData # re-reads tables; verify-on-read (§7)

    # ML ops
    def split(self, *, by="study_id", train, val, test, seed, mode="group") -> SplitManifest   # [C5]
    def check(self, adapter: TaskAdapter, *, deep: bool = False) -> PreflightReport             # [C6]
    def build(self, adapter: TaskAdapter, *, split=None, materialize=False, out=None, verify=True) -> TaskView  # [C7]
```

`load()` reconstructs the **manifest** (membership + metadata) with no catalog access; it does NOT fetch
object data. Any method that reads object bytes (`to_anndata`, `build` with `materialize=True`) performs
**verify-on-read** (§7) unless `verify=False`. `frozen=True` prevents attribute rebinding but NOT in-place
mutation of the `members` DataFrame — callers must treat `members` as read-only; the hash↔content
invariant is not machine-enforced against DataFrame mutation (document this; expose `to_df()` copies for
any caller that needs to edit).

## 6. Data flow

```
Catalog.query(...)            → SpatialDataCohort(df, filter, source)
SpatialDataCohort.freeze()    → object_metadata(all uids, parallel, metadata-only)
                              → resolve member table (catalog row + lamin_version + fingerprint + obj-meta)
                              → compute hash → FrozenCohort
FrozenCohort.write(path)      → path.json (+ path.members.parquet when large)
FrozenCohort.load(path)       → FrozenCohort            (no catalog touched)
FrozenCohort.split(...)       → SplitManifest(parent_hash=hash, assignments)
FrozenCohort.check(adapter)   → PreflightReport
FrozenCohort.build(adapter, split, materialize, out) → TaskView | writes out/ + provenance.json
```

## 7. Freeze file format & hash

**File:** one JSON:
```json
{
  "schema_version": "1",
  "created_at": "2026-10-07T…Z",
  "source": {"url": "…catalog.parquet", "version": "2026-10-05"},
  "filter": {"organism": "human", "technology": "Visium HD", "...": "..."},
  "hash": "sha256:…",
  "n_members": 83,
  "members": [{"uid": "…", "lamin_version": "…", "zarr_url": "…", "fingerprint": "sha256:…",
               "study_id": "geo:GSE…", "default_table": "table", "n_obs": 12345, "...catalog/obj-meta...": "…"}]
}
```
When `n_members` is large, write `members` to a `*.members.parquet` sidecar and store a
`{"members_ref": "<file>.members.parquet", "members_sha256": "…"}` pointer in the JSON instead.

**Hash** = sha256 over canonical (sorted-key, no-whitespace) JSON of `{filter, members}` where `members`
is sorted by `uid` and each member reduced to `{uid, lamin_version, fingerprint}`. **Excludes**
`created_at`. → freezing the same query against the same catalog version twice gives the same hash;
membership or version change ⇒ hash change.

**`fingerprint`** (per object, from C2): sha256 over the object's metadata (element inventory + array
shapes/chunks + table list + lamin_version).

**Verify-on-read (enforced, not advisory).** `load()` + manifest inspection touch no object data. The
moment a `FrozenCohort` reads object bytes (`to_anndata`, `build(materialize=True)`), it re-fetches each
member's metadata and recomputes `fingerprint`; a mismatch against the manifest (or a changed
`lamin_version`) **raises** `CohortDriftError`, naming the offending uids. `verify=False` opts out
explicitly for speed. This is what makes "reconstruct the frozen cohort" real: without it, a store
republished at the same URL would be read silently under the original hash. Honest scope: verification is
at **metadata + shape** granularity (what C2 captures), not chunk-byte granularity — state this in the
docs rather than claiming byte-level immutability.

## 8. Component specs

### 8.1 `objectmeta.py` [C2]
`fetch_object_metadata(zarr_url, *, storage_options) -> ObjectMeta` reads ONLY zarr metadata
(`.zgroup`/`.zattrs`/`.zarray`, SpatialData element inventory, table names, obs/var counts, lamin
version) via `fsspec` — never array chunks. `fetch_many(urls, workers) -> list[ObjectMeta]` parallelizes.
Returns a `fingerprint` per object. Depends on C0's finding for the version field.

**Partial-failure semantics (all-or-nothing).** `freeze()` fans this out over up to ~600 remote objects.
If ANY member's metadata fetch fails (timeout, 403, missing store), `freeze()` **raises**, listing every
failed uid — it never drops a member or fills partial metadata, because a silently-incomplete freeze
produces a valid-looking hash for the wrong membership. This mirrors `ManifestEntry.status` discipline but
at freeze there is no "partial success" state: a cohort is frozen completely or not at all. (Transient
failures are the caller's to retry; `fetch_many` may expose a bounded retry, but the freeze contract is
binary.) Verify-on-read (§7) reuses this same fetch path.

### 8.2 `concat.py` [C4]
`concat_tables(members, *, table=None, storage_options) -> AnnData`: for each member, open the
`tables/<default_table or table>` group lazily and `anndata.concat` (join="outer" on var, label by uid).
Raise a clear error if the requested table is absent for a member. Images/shapes stay remote. Records
the table used per member (for provenance).

### 8.3 `split.py` [C5]
```python
@dataclass(frozen=True)
class SplitManifest:
    parent_hash: str; by: str; seed: int; ratios: dict; mode: str
    assignments: dict[str, str]   # uid → "train"|"val"|"test"  (or group label in mode="annotate")
    @classmethod
    def load(cls, path) -> SplitManifest
    def write(self, path) -> Path
```
`mode="group"` (default): assign whole groups to folds by `ratios` → no group spans two folds.
`mode="annotate"`: attach the group label per uid without partitioning (consumer defines holdout). `apply`
against a `FrozenCohort` whose `hash` ≠ `parent_hash` raises.

**Determinism (must survive version/env skew).** The group keys (`by`, missing → `object:<uid>`) are
**sorted lexicographically before seeding** — never fed straight from a `groupby`/`set` whose order can
vary across pandas/Python versions. Shuffle via `numpy.random.default_rng(seed)` (not global state) over
that sorted list. → identical `(seed, ratios, membership)` gives an identical split on any machine; the
C5 acceptance test asserts this against a frozen expected assignment.

**Small-cohort edges (our reality: 184 singleton studies).** Whole-group assignment over few groups can
round a non-zero ratio to an empty fold. The rule: each fold with a **non-zero ratio is guaranteed ≥1
group**; if the cohort has fewer groups than non-zero folds, `split()` **raises** with a message naming
the group count vs the requested folds (never silently emits an empty `val`/`test`). Document the rounding
(largest-remainder) so counts are predictable.

### 8.4 `adapter.py` [C6/C7]
```python
class TaskAdapter(Protocol):
    def requirements(self) -> Requirements: ...               # declares needed elements/modalities
    def validate_meta(self, meta: ObjectMeta) -> list[Issue]: ...   # cheap, ADVISORY (metadata only)
    def validate(self, sdata: SpatialData) -> list[Issue]: ...      # AUTHORITATIVE ([] = compatible)
    def build(self, sdata: SpatialData) -> Any: ...            # task representation (format owned by adapter)

@dataclass
class PreflightReport:
    total: int; compatible: int; excluded: dict[str, list[str]]   # reason → uids
    tier: str   # "advisory" (metadata only) | "authoritative" (deep)
    def __repr__(self) -> str   # renders "83 candidate / 79 compatible / 2 missing H&E / …" + the tier

def check(frozen, adapter, *, deep=False) -> PreflightReport:
    # deep=False: adapter.validate_meta(member ObjectMeta)  — cheap, tier="advisory"
    # deep=True:  adapter.validate(open_sdata(member.zarr_url)) — authoritative, tier="authoritative"

class TaskView:                    # lazy: per-member build thunks + metadata; iterable/indexable
    def __iter__(self): ...        # a user's own dataloader can wrap this

def build(frozen, adapter, *, split=None, materialize=False, out=None) -> TaskView:
    # materialize=False → return TaskView (no heavy data pulled)
    # materialize=True  → for each member: adapter.build(open_sdata(url)); write to out/; write provenance.json
```
`provenance.json`: `{cohort_hash, split_hash|null, adapter:{name, version, config}, sddb_version,
output_schema, created_at, members:[{uid, table, output_path}]}`.

**Evidence rule:** any compatibility count cited as paper evidence MUST come from a `deep=True`
(authoritative) pass — the advisory tier can over-count "compatible" because `validate_meta` can't see
inside the store (e.g. a valid image↔expression transform). `PreflightReport.tier` records which ran so a
number is never quoted without its provenance.

## 9. Cross-repo asks to the engine producer (parallel, non-blocking)

- **E1** backfill `study_id` for the 72 missing (reduces singleton fallback).
- **E2** add `donor_id` where recoverable (enables donor-aware holdout later).
- **E3** expose a stable **lamin version** column (so freeze records it unambiguously, not only via `uid`).

Client builds entirely on today's catalog meanwhile; missing fields read as `"unknown"`.

## 10. Work packages & acceptance criteria

| WP | Deliverable | Acceptance |
|----|-------------|------------|
| **C0** (spike) | lamin version pinning + mutability | determine whether `zarr_url` is version-qualified/immutable or a mutable pointer; identify the version field. Output drives the verify-on-read design (§7). Throwaway. |
| **C1** | `SpatialDataCohort` + inspection; rename `Results` | `query().coverage()`/`groupby("study_id")` on the real 608-row catalog; unknown bucket + 72 singleton studies surfaced; all renamed call-sites + tests pass. |
| **C2** | `objectmeta.py` metadata-only fetch | fetch for a few real objects incl. a multi-table one; version + fingerprint recorded; no array bytes pulled; all-or-nothing on partial failure (raises, lists failed uids). |
| **C3** | `freeze()`/`FrozenCohort.load`/hash + verify-on-read | freeze→write→load reproduces membership+metadata with no catalog; hash deterministic and sensitive to membership/version; a data read with a mutated fingerprint raises `CohortDriftError`; `verify=False` bypasses. |
| **C4** | lazy `to_anndata()` | concat across a small multi-study cohort without full-store download; table used recorded. |
| **C5** | `SplitManifest` | same seed→same split (asserted against a frozen expected assignment, env-independent via sorted keys + `default_rng`); group mode leaks no study; annotate mode labels only; a cohort with fewer groups than non-zero folds raises (no empty fold); reload; wrong `parent_hash` raises. |
| **C6** | `TaskAdapter` + `check()` | in-test adapter with `validate_meta` + `validate`; `check()` returns "N compatible / M excluded (reasons)" + `tier`; `deep=True` opens stores and is authoritative. |
| **C7** | `build(materialize=)` + provenance | build→materialize a toy adapter; `provenance.json` complete and correct. |

**Sequence:** `C0 → C1 → C2 → C3 → {C4, C5} → C6 → C7`. **Milestone 1** (prove in isolation): C0–C3 (+C4).
**Milestone 2:** C5–C7. E1/E2/E3 run in parallel.

## 11. Testing

Offline units against the existing fixture catalog (`tests/_fixtures.py`) + a tiny local zarr fixture
(1–2 tables) for objectmeta/concat/freeze. Determinism asserts for hash (C3) and split (C5, against a
frozen expected assignment). **Verify-on-read:** mutate a fixture store's metadata after freeze → a data
read raises `CohortDriftError`; `verify=False` reads anyway. **Partial-failure:** point one member at a
missing store → `freeze()` raises and names it. **Split edges:** fewer groups than non-zero folds raises.
`@pytest.mark.network` tests for real-catalog freeze and lazy concat (opt-in, deselected by default).
mypy-strict + ruff clean; sphinx build clean.

## 12. Out of scope (separate specs)

CIVET `CivetAdapter` + 5-object equivalence (CIVET repo); reference layer / `sddb.reference()` (strategy
WP6); engine producer changes E1–E3; multi-table joins beyond `default_table`; full lazy SpatialData
concat (blocked upstream); named DB releases + DOIs; promoting a frozen cohort to a hosted collection.
