"""Metadata-only per-object fetch + drift fingerprint (never reads array chunks)."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from sddb.remote import _open_group, elements

_ROOT_KEYS = ("sddb_provenance", "spatialdata_attrs")


class ObjectMetaError(RuntimeError):
    """One or more object-metadata fetches failed; `failed` maps uid -> error string."""

    def __init__(self, failed: dict[str, str]) -> None:
        self.failed = failed
        super().__init__(f"metadata fetch failed for {len(failed)} object(s): {sorted(failed)}")


@dataclass(frozen=True)
class ObjectMeta:
    """Metadata snapshot of one object; `fingerprint` is the drift guard."""

    uid: str
    zarr_url: str
    lamin_version: str | None
    elements: dict[str, dict[str, Any]]
    tables: tuple[str, ...]
    fingerprint: str


def compute_fingerprint(elements: dict[str, Any], tables: tuple[str, ...], lamin_version: str | None) -> str:
    """Stable sha256 over element inventory, tables and version."""
    payload = json.dumps(
        {
            "elements": {k: elements[k] for k in sorted(elements)},
            "tables": sorted(tables),
            "lamin_version": lamin_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _root_attrs(zarr_url: str) -> dict[str, Any]:
    """Root-group provenance attrs (one metadata read); absent keys are omitted, read errors propagate."""
    a = dict(_open_group(zarr_url).attrs)
    return {k: a[k] for k in _ROOT_KEYS if k in a}


def fetch_object_metadata(uid: str, zarr_url: str) -> ObjectMeta:
    """Read one object's zarr metadata (no chunks) and fingerprint it, root provenance included."""
    el = elements(zarr_url)  # raises if the store cannot be opened; metadata only
    root = _root_attrs(zarr_url)
    tables = tuple(sorted(k.split("/", 1)[1] for k in el if k.startswith("tables/")))
    ver = uid[16:] or None  # lamin version ordinal suffix (bookkeeping only)
    fp = compute_fingerprint({**el, "__root__": root}, tables, ver)
    return ObjectMeta(uid, zarr_url, ver, el, tables, fp)


def fetch_many(pairs: list[tuple[str, str]], *, workers: int = 8) -> list[ObjectMeta]:
    """Fetch many objects concurrently; all-or-nothing (raises ObjectMetaError)."""
    out: dict[str, ObjectMeta] = {}
    failed: dict[str, str] = {}

    def one(p: tuple[str, str]) -> None:
        uid, url = p
        try:
            out[uid] = fetch_object_metadata(uid, url)
        except Exception as e:  # aggregated and re-raised as ObjectMetaError
            failed[uid] = f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, pairs))
    if failed:
        raise ObjectMetaError(failed)
    return [out[uid] for uid, _ in pairs]
