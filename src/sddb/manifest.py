"""Cohort identity record (CohortManifest, cohort_hash, CohortDriftError, CatalogChangedError) and resumable cohort download with a JSON manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec
import pandas as pd

from sddb._progress import _progress, _sizes
from sddb.dataset import Source
from sddb.remote import storage_options

if TYPE_CHECKING:
    from sddb.cohort import SpatialDataCohort


SCHEMA_VERSION = "2"
_SIDECAR_THRESHOLD = 200  # members beyond this go to a parquet sidecar


def cohort_hash(filter: dict[str, Any], uids: Iterable[Any]) -> str:
    """sha256 over the canonical filter and the sorted full lamin uids (version is baked into the uid)."""
    payload = json.dumps(
        {"filter": filter, "uids": sorted(str(u) for u in uids)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class CohortDriftError(RuntimeError):
    """A pinned cohort member no longer resolves; `drifted` maps uid -> reason."""

    def __init__(self, drifted: dict[str, str]) -> None:
        self.drifted = drifted
        super().__init__(f"{len(drifted)} member(s) no longer resolve: {sorted(drifted)}")


class CatalogChangedError(RuntimeError):
    """The pinned catalog's bytes differ from the sha256 recorded when the cohort was frozen."""

    def __init__(self, url: str, expected: str, actual: str) -> None:
        self.url = url
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"catalog at {url} was modified after the cohort was frozen: expected {expected}, got {actual}"
        )


def _check_catalog_sha(url: str, expected: str, actual: str) -> None:
    """Compare a cohort's recorded catalog hash with the current one.

    A mismatch raises :class:`CatalogChangedError` for a pinned catalog (any url but the rolling
    ``catalog-latest`` one) and only warns for ``catalog-latest``, which is expected to move.
    """
    from sddb.catalog import _ROLLING_SUFFIX

    if expected == actual:
        return
    if not url.endswith(_ROLLING_SUFFIX):
        raise CatalogChangedError(url, expected, actual)
    warnings.warn(
        f"catalog-latest changed since the cohort was frozen ({expected} -> {actual}); "
        "pin version= for a reproducible cohort",
        UserWarning,
        stacklevel=3,  # user -> verify/from_manifest -> here
    )


@dataclass(frozen=True, eq=False)
class CohortManifest:
    """Serializable cohort identity: filter, member snapshot (pinned full uids) and a deterministic hash."""

    schema_version: str
    created_at: str
    source: Source
    filter: dict[str, Any]
    hash: str
    members: pd.DataFrame  # read-only by contract

    def write(self, path: str | Path) -> Path:
        """Write JSON to ``path``; members spill to a parquet sidecar when large. Returns the path."""
        p = Path(path)
        head: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source": {"url": self.source.url, "version": self.source.version, "sha256": self.source.sha256},
            "filter": self.filter,
            "hash": self.hash,
            "n_members": len(self.members),
        }
        if len(self.members) > _SIDECAR_THRESHOLD:
            side = p.with_suffix(".members.parquet")
            self.members.to_parquet(side)
            head["members_ref"] = side.name
        else:
            head["members_columns"] = list(self.members.columns)
            head["members"] = json.loads(self.members.to_json(orient="records"))
        p.write_text(json.dumps(head, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> CohortManifest:
        """Read a manifest written by :meth:`write`."""
        p = Path(path)
        head = json.loads(p.read_text())
        if "members_ref" in head:
            members = pd.read_parquet(p.parent / head["members_ref"])
        else:
            members = pd.DataFrame(head["members"], columns=head.get("members_columns"))
        raw = head["source"]
        src = Source(url=raw["url"], version=raw["version"], sha256=raw.get("sha256"))  # sha256 absent in old manifests
        return cls(head["schema_version"], head["created_at"], src, head["filter"], head["hash"], members)


@dataclass
class ManifestEntry:
    """One downloaded store."""

    uid: str
    zarr_url: str
    dest: str
    size_bytes: int | None
    status: str  # "complete" | "failed" | "partial" (only ``elements`` were copied)
    sha256: str | None
    error: str | None = None  # why a failed download failed ("<ExceptionType>: <message>")
    elements: list[str] | None = None  # the requested element paths of a partial copy


@dataclass
class Manifest:
    """What was downloaded, where, and from which catalog (``catalog_version`` is None when not pinned)."""

    generated_at: str
    catalog_version: str | None
    entries: list[ManifestEntry]

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        """Read a manifest written by :meth:`write`."""
        raw = json.loads(Path(path).read_text())
        return cls(raw["generated_at"], raw.get("catalog_version"), [ManifestEntry(**e) for e in raw["entries"]])

    def refetch(self, dest: str | Path, *, workers: int = 4) -> Manifest:
        """Re-download exactly the stores recorded in this manifest to ``dest`` (reproducible re-fetch).

        Uses the recorded ``zarr_url`` of each entry, not the current catalog, and keeps the recorded
        ``catalog_version``. Completed stores already in ``dest`` are skipped. A partial entry copies its
        recorded ``elements`` again, not the whole store. Only URLs are replayed: a store republished at
        the same URL is not detected.
        """
        root = Path(dest)
        pairs = [(e.uid, e.zarr_url) for e in self.entries if e.status != "partial"]
        _fetch_all(pairs, root, self.catalog_version, workers)
        for e in self.entries:
            if e.status == "partial":
                _localize_all([(e.uid, e.zarr_url)], e.elements or [], root, workers)
        return Manifest.read(root / "manifest.json")

    def write(self, path: str | Path) -> Path:
        """Write the manifest as JSON and return the path."""
        p = Path(path)
        doc = asdict(self)
        for e in doc["entries"]:
            if e["elements"] is None:
                del e["elements"]  # so clients without the field still read manifests of complete stores
        p.write_text(json.dumps(doc, indent=2))
        return p


def _copy_store(zarr_url: str, dest: Path) -> None:
    """Copy a (remote or local) store to ``dest`` (which must not exist)."""
    fs, root = fsspec.core.url_to_fs(zarr_url, **storage_options(zarr_url))
    if not fs.exists(root):
        raise FileNotFoundError(zarr_url)
    dest.mkdir(parents=True)
    fs.get(root.rstrip("/") + "/", str(dest) + "/", recursive=True)


def _digest(store: Path) -> tuple[int, str]:
    """Return (total bytes, sha256 over sorted relative paths + sizes) of a local store."""
    h, total = hashlib.sha256(), 0
    for f in sorted(p for p in store.rglob("*") if p.is_file()):
        size = f.stat().st_size
        total += size
        h.update(f"{f.relative_to(store).as_posix()}:{size}\n".encode())
    return total, h.hexdigest()


class ManifestVersionMismatch(ValueError):
    """The catalog version differs from the one recorded in an existing manifest."""


def _fetch_all(
    pairs: list[tuple[str, str]],
    root: Path,
    catalog_version: str | None,
    workers: int,
    sizes: dict[str, int] | None = None,
) -> Manifest:
    """Copy each ``(uid, zarr_url)`` to ``root/<uid>.zarr`` (skipping completed ones) and write the manifest."""
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    done: dict[str, ManifestEntry] = {}
    if mpath.exists():
        done = {e.uid: e for e in Manifest.read(mpath).entries if e.status == "complete"}
    manifest = Manifest(datetime.now(UTC).isoformat(), catalog_version, [])
    lock = threading.Lock()
    by_uid: dict[str, ManifestEntry] = {}

    def save() -> None:
        manifest.entries = [by_uid[u] for u, _ in pairs if u in by_uid]
        manifest.write(mpath)

    def have(uid: str, url: str) -> bool:
        prev = done.get(uid)
        return prev is not None and prev.zarr_url == url and (root / f"{uid}.zarr").is_dir()

    todo = {u for u, url in pairs if not have(u, url)}
    sizes = sizes or {}

    def fetch(uid: str, url: str) -> None:
        final = root / f"{uid}.zarr"
        if have(uid, url):
            entry = done[uid]
        else:
            part = root / f"{uid}.zarr.part"
            for stale in (part, final):
                if stale.exists():
                    shutil.rmtree(stale)
            try:
                _copy_store(url, part)
                size, sha = _digest(part)
                part.rename(final)
                entry = ManifestEntry(uid, url, str(final), size, "complete", sha)
            except Exception as err:
                shutil.rmtree(part, ignore_errors=True)
                entry = ManifestEntry(uid, url, str(final), None, "failed", None, f"{type(err).__name__}: {err}")
        with lock:
            by_uid[uid] = entry
            save()
            if uid in todo and entry.status == "complete":
                bar.update(sizes.get(uid, 0))

    with _progress(len(todo), sum(sizes.get(u, 0) for u in todo)) as bar:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(lambda p: fetch(*p), pairs))
    save()
    return manifest


def _localize_all(pairs: list[tuple[str, str]], elements: list[str], root: Path, workers: int) -> list[Path]:
    """Copy only ``elements`` of each ``(uid, zarr_url)`` to ``root/<uid>.zarr``, recorded as ``"partial"`` in the manifest.

    A complete store already in ``root`` is opened as it is. A partial copy is reused when it holds the requested
    elements; otherwise its recorded elements plus the requested ones are copied again.
    """
    from sddb.remote import _localize_to_cache, _missing

    if not elements:
        raise ValueError("pass element paths in `elements`, or None for the whole store")
    if bad := [p for p in elements if not p.partition("/")[2]]:
        raise ValueError(f"element path must be 'kind/name', got {bad[0]!r}")
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    prev = {e.uid: e for e in Manifest.read(mpath).entries} if mpath.exists() else {}
    full = reusable(root)

    def old(uid: str, url: str) -> list[str] | None:
        """The recorded elements of a partial copy of ``url`` in ``root``, else None."""
        e = prev.get(uid)
        if e is None or e.status != "partial" or e.zarr_url != url:
            return None
        return e.elements or []

    def want(uid: str, url: str) -> list[str]:
        return sorted({*elements, *(old(uid, url) or [])})

    todo = {
        u
        for u, url in pairs
        if full.get(u) != url and (old(u, url) is None or _missing(root / f"{u}.zarr", want(u, url)))
    }
    order = {u: i for i, (u, _) in enumerate(pairs)}  # entries in member order, as download() writes them
    lock = threading.Lock()

    def record(uid: str, entry: ManifestEntry | None) -> None:
        """Replace (or with ``None`` drop) the manifest entry for ``uid``."""
        with lock:
            m = Manifest.read(mpath) if mpath.exists() else Manifest("", None, [])
            m.generated_at = datetime.now(UTC).isoformat()
            kept = [e for e in m.entries if e.uid != uid] + ([entry] if entry else [])
            m.entries = sorted(kept, key=lambda e: order.get(e.uid, -1))
            m.write(mpath)

    def local(uid: str, url: str) -> Path:
        final = root / f"{uid}.zarr"
        if uid not in todo:
            return final
        if old(uid, url) is None:  # not a copy this manifest vouches for: unlist it before removing it
            if uid in prev:
                record(uid, None)
            shutil.rmtree(final, ignore_errors=True)
        paths = want(uid, url)
        _localize_to_cache(url, paths, root, final)
        size, sha = _digest(final)
        record(uid, ManifestEntry(uid, url, str(final), size, "partial", sha, elements=paths))
        bar.update(size)
        return final

    with _progress(len(todo), 0) as bar:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(lambda p: local(*p), pairs))


def plan_sizes(results: SpatialDataCohort) -> list[tuple[str, int]]:
    """Per-store ``(uid, total bytes)`` from a LIST/``du`` on each store — no data is copied."""
    out: list[tuple[str, int]] = []
    for ds in results:
        fs, root = fsspec.core.url_to_fs(ds.zarr_url, **storage_options(ds.zarr_url))
        out.append((ds.uid, int(fs.du(root))))
    return out


def _store_exists(zarr_url: str) -> bool:
    """Whether a store resolves on its filesystem (anon S3 or local).

    Only a confirmed non-existence returns False; transient/auth/network errors propagate.
    """
    from sddb.remote import _open_fs

    try:
        fs, root = _open_fs(zarr_url)
        return bool(fs.exists(root))
    except FileNotFoundError:
        return False


def _latest_for_stem(source: Source, stem: str) -> tuple[str, str] | None:
    """Resolve the current ``(uid, zarr_url)`` for a uid stem from the cohort's source catalog.

    Returns None if the catalog is unreachable or the stem is absent.
    """
    from sddb.catalog import Catalog

    if not source.url:
        return None
    try:
        df = Catalog(source.url).to_df()
    except Exception:
        return None
    hit = df[df["uid"].astype("string").str[:16] == stem]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return str(row["uid"]), str(row["zarr_url"])


def reusable(root: str | Path) -> dict[str, str]:
    """``{uid: zarr_url}`` of the stores that ``download`` will not fetch again into ``root``.

    These are the entries the manifest in ``root`` lists as complete whose ``<uid>.zarr`` is still there. A store
    is reused only while its catalog ``zarr_url`` stays the same.
    """
    root = Path(root)
    mpath = root / "manifest.json"
    if not mpath.exists():
        return {}
    entries = Manifest.read(mpath).entries
    return {e.uid: e.zarr_url for e in entries if e.status == "complete" and (root / f"{e.uid}.zarr").is_dir()}


def _check_pin(
    mpath: Path,
    current: str | None,
    allow_version_change: bool,
    *,
    hint: str = "pass allow_version_change=True to overwrite",
) -> bool:
    """Whether the manifest at ``mpath`` is pinned to ``current``; raise if it is pinned to another version.

    A mismatch raises :class:`ManifestVersionMismatch` unless ``allow_version_change`` (then False: not pinned).
    """
    recorded = Manifest.read(mpath).catalog_version if mpath.exists() else None
    if recorded is None or recorded == current:
        return recorded is not None
    if not allow_version_change:
        raise ManifestVersionMismatch(
            f"manifest at {mpath} is pinned to catalog version {recorded!r}, "
            f"but the current catalog version is {current!r}; {hint}"
        )
    return False


def download(
    results: SpatialDataCohort,
    dest: str | Path,
    *,
    workers: int = 4,
    pin_versions: bool = False,
    allow_version_change: bool = False,
    strict: bool = False,
) -> Manifest:
    """Download every dataset's store to ``dest/<uid>.zarr`` and write ``dest/manifest.json``.

    Resumable: entries already ``complete`` in an existing manifest (with the store present) are
    skipped; failed/partial ones are re-fetched. Stores are copied to ``<uid>.zarr.part`` and
    renamed on success. ``pin_versions`` records the catalog version in the manifest; an existing
    manifest pinned to a different catalog version raises :class:`ManifestVersionMismatch` unless
    ``allow_version_change`` is True (then it is overwritten). Only the catalog version is pinned:
    per-dataset byte-level pinning across republishes is not implemented. The sha256 covers store
    file paths and sizes, not contents.

    Each store's liveness is checked first: a missing pinned store raises :class:`CohortDriftError`
    when ``strict``, else falls back to the latest version (via the source catalog) with a warning.
    """
    root = Path(dest)
    current = results._source.version
    mpath = root / "manifest.json"
    pinned = _check_pin(mpath, current, allow_version_change) or pin_versions
    pairs: list[tuple[str, str]] = []
    missing: dict[str, str] = {}
    present = reusable(root)
    for d in results:
        uid, url = str(d.uid), str(d.zarr_url)
        if present.get(uid) == url:
            pairs.append((uid, url))  # already complete locally: no liveness check, no fetch
            continue
        try:
            alive = _store_exists(url)
        except Exception as err:
            raise RuntimeError(f"could not determine liveness of {uid} store at {url}: {err}") from err
        if alive:
            pairs.append((uid, url))
            continue
        if strict:
            missing[uid] = f"pinned store missing: {url}"
            continue
        latest = _latest_for_stem(results._source, uid[:16])
        if latest is None:
            missing[uid] = f"pinned store missing and no fallback resolvable: {url}"
            continue
        warnings.warn(
            f"member {uid}: pinned store missing; FALLING BACK to latest {latest[0]} "
            f"({latest[1]}) - bytes DIFFER from the pinned version",
            stacklevel=3,  # user -> SpatialDataCohort.download -> here
        )
        pairs.append((uid, latest[1]))
    if missing:
        raise CohortDriftError(missing)
    sizes = dict(zip(results._df["uid"].astype(str), _sizes(results._df), strict=True))
    return _fetch_all(pairs, root, current if pinned else None, workers, sizes)
