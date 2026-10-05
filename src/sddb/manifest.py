"""Resumable cohort download with a JSON manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec

from sddb.remote import storage_options

if TYPE_CHECKING:
    from sddb.dataset import Results


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


def download(
    results: Results,
    dest: str | Path,
    *,
    workers: int = 4,
    pin_versions: bool = False,
    allow_version_change: bool = False,
) -> Manifest:
    """Download every dataset's store to ``dest/<uid>.zarr`` and write ``dest/manifest.json``.

    Resumable: entries already ``complete`` in an existing manifest (with the store present) are
    skipped; failed/partial ones are re-fetched. Stores are copied to ``<uid>.zarr.part`` and
    renamed on success. ``pin_versions`` records the catalog version in the manifest; an existing
    manifest pinned to a different catalog version raises :class:`ManifestVersionMismatch` unless
    ``allow_version_change`` is True (then it is overwritten). Only the catalog version is pinned:
    per-dataset byte-level pinning across republishes is not implemented. The sha256 covers store
    file paths and sizes, not contents.
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
    pairs = [(str(d.uid), str(d.zarr_url)) for d in results]
    return _fetch_all(pairs, root, current if pinned else None, workers)
