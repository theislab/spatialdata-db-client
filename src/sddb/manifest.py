"""Cohort identity record (CohortManifest, cohort_hash, CohortDriftError) and resumable cohort download with a JSON manifest."""

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
            "source": {"url": self.source.url, "version": self.source.version},
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
        src = Source(url=head["source"]["url"], version=head["source"]["version"])
        return cls(head["schema_version"], head["created_at"], src, head["filter"], head["hash"], members)


@dataclass
class ManifestEntry:
    """One downloaded store."""

    uid: str
    zarr_url: str
    dest: str
    size_bytes: int | None
    status: str  # "complete" | "failed"
    sha256: str | None
    error: str | None = None  # why a failed download failed ("<ExceptionType>: <message>")


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
        ``catalog_version``. Completed stores already in ``dest`` are skipped. Only URLs are replayed:
        a store republished at the same URL is not detected.
        """
        pairs = [(e.uid, e.zarr_url) for e in self.entries]
        return _fetch_all(pairs, Path(dest), self.catalog_version, workers)

    def write(self, path: str | Path) -> Path:
        """Write the manifest as JSON and return the path."""
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2))
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


def _fetch_all(pairs: list[tuple[str, str]], root: Path, catalog_version: str | None, workers: int) -> Manifest:
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

    def fetch(uid: str, url: str) -> None:
        final = root / f"{uid}.zarr"
        prev = done.get(uid)
        if prev is not None and prev.zarr_url == url and final.is_dir():
            entry = prev
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

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(lambda p: fetch(*p), pairs))
    save()
    return manifest


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
    pinned = pin_versions
    if mpath.exists():
        recorded = Manifest.read(mpath).catalog_version
        if recorded is not None and recorded != current:
            if not allow_version_change:
                raise ManifestVersionMismatch(
                    f"manifest at {mpath} is pinned to catalog version {recorded!r}, "
                    f"but the current catalog version is {current!r}; "
                    "pass allow_version_change=True to overwrite"
                )
        elif recorded is not None:
            pinned = True
    pairs: list[tuple[str, str]] = []
    missing: dict[str, str] = {}
    complete: dict[str, ManifestEntry] = {}
    if mpath.exists():
        complete = {e.uid: e for e in Manifest.read(mpath).entries if e.status == "complete"}
    for d in results:
        uid, url = str(d.uid), str(d.zarr_url)
        prev = complete.get(uid)
        if prev is not None and prev.zarr_url == url and (root / f"{uid}.zarr").is_dir():
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
    return _fetch_all(pairs, root, current if pinned else None, workers)
