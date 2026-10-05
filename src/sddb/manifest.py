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


@dataclass
class Manifest:
    """What was downloaded, where, and from which catalog."""

    generated_at: str
    catalog_version: str | None
    entries: list[ManifestEntry]

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        """Read a manifest written by :meth:`write`."""
        raw = json.loads(Path(path).read_text())
        return cls(raw["generated_at"], raw.get("catalog_version"), [ManifestEntry(**e) for e in raw["entries"]])

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


def download(results: Results, dest: str | Path, *, workers: int = 4, pin_versions: bool = False) -> Manifest:
    """Download every dataset's store to ``dest/<uid>.zarr`` and write ``dest/manifest.json``.

    Resumable: entries already ``complete`` in an existing manifest (with the store present) are
    skipped; failed/partial ones are re-fetched. Stores are copied to ``<uid>.zarr.part`` and
    renamed on success. ``pin_versions`` records the catalog version in the manifest (per-dataset
    pinning across republishes is not available yet). The sha256 covers store file paths and sizes,
    not contents.
    """
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    done: dict[str, ManifestEntry] = {}
    if mpath.exists():
        done = {e.uid: e for e in Manifest.read(mpath).entries if e.status == "complete"}
    manifest = Manifest(datetime.now(UTC).isoformat(), results._source.version if pin_versions else None, [])
    lock = threading.Lock()
    by_uid: dict[str, ManifestEntry] = {}

    def save() -> None:
        order = [d.uid for d in results]
        manifest.entries = [by_uid[u] for u in order if u in by_uid]
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
            except Exception:
                shutil.rmtree(part, ignore_errors=True)
                entry = ManifestEntry(uid, url, str(final), None, "failed", None)
        with lock:
            by_uid[uid] = entry
            save()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(lambda d: fetch(str(d.uid), str(d.zarr_url)), list(results)))
    save()
    return manifest
