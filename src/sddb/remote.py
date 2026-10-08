"""Anonymous remote read primitives for SpatialData zarr stores."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import fsspec
import zarr

from sddb._cache import cache_dir as _cache_dir

if TYPE_CHECKING:
    from spatialdata import SpatialData

_ELEMENT_KINDS = ("images", "labels", "points", "shapes", "tables")


def storage_options(url: str) -> dict[str, Any]:
    """Return fsspec storage options for anonymous read.

    Parameters
    ----------
    url
        ``s3://`` (anonymous), ``http(s)://`` or a local path.

    Returns
    -------
    ``{"anon": True}`` for S3, otherwise ``{}``.
    """
    return {"anon": True} if url.startswith("s3://") else {}


_PROXY = "https://lamin.ai/storage/s3/"


def proxy_url(url: str) -> str:
    """Rewrite ``s3://bucket/key`` to the LaminHub CORS proxy URL; other URLs pass through."""
    return _PROXY + url[len("s3://") :] if url.startswith("s3://") else url


def _is_remote(url: str) -> bool:
    return url.startswith(("s3://", "http://", "https://"))


def open_sdata(
    zarr_url: str,
    *,
    lazy: bool = True,
    elements: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> SpatialData:
    """Open a SpatialData zarr from a URL or local path, anonymously.

    Parameters
    ----------
    zarr_url
        Store URL or local path.
    lazy
        If True, open a *local* store in place (dask-backed). Not supported for remote URLs
        (``s3://``, ``http(s)://``): spatialdata's remote ``read_zarr`` is broken upstream. If False,
        copy the store (remote or local) into the cache directory and open the local copy.
    elements
        Element paths (``"<kind>/<name>"``) to localize; only these are copied and opened. ``None``
        opens the whole store. Requires ``lazy=False``.
    cache_dir
        Cache directory override (only used when ``lazy=False``).

    Returns
    -------
    The opened SpatialData object.

    Raises
    ------
    NotImplementedError
        If ``zarr_url`` is remote and ``lazy=True``.
    FileNotFoundError
        If the store cannot be opened; the message names ``zarr_url``.
    """
    from spatialdata import read_zarr

    if elements is not None:
        if lazy:
            raise NotImplementedError("elements= requires lazy=False (a partial store is always copied)")
        dest = _localize_to_cache(zarr_url, list(elements), _cache_dir(cache_dir))
        return read_zarr(dest)
    if lazy and _is_remote(zarr_url):
        raise NotImplementedError(
            f"Remote lazy/partial open is not supported ({zarr_url!r}): blocked by a spatialdata upstream "
            "limitation in remote read_zarr. Use lazy=False to download the store to the cache and open it, "
            "or elements() to inspect it without downloading."
        )
    try:
        if not lazy:
            local = _copy_to_cache(zarr_url, _cache_dir(cache_dir))
            return read_zarr(local)
        return read_zarr(Path(zarr_url))
    except FileNotFoundError:
        raise
    except Exception as e:
        raise FileNotFoundError(f"Cannot open SpatialData zarr at {zarr_url!r}: {e}") from e


def _open_group(zarr_url: str) -> zarr.Group:
    """Open a store read-only as a ``zarr.Group`` (anonymous for S3)."""
    return zarr.open_group(zarr_url, mode="r", storage_options=storage_options(zarr_url) or None)


def _copy_to_cache(zarr_url: str, dest_root: Path) -> Path:
    """Copy a (remote or local) store to ``dest_root/<basename>.zarr`` and return the path."""
    name = zarr_url.rstrip("/").rsplit("/", 1)[-1]
    if not name.endswith(".zarr"):
        name += ".zarr"
    dest = dest_root / name
    fs, root = fsspec.core.url_to_fs(zarr_url, **storage_options(zarr_url))
    if not fs.exists(root):
        raise FileNotFoundError(zarr_url)
    dest.mkdir(parents=True, exist_ok=True)
    fs.get(root.rstrip("/") + "/", str(dest) + "/", recursive=True)
    return dest


def _prune_consolidated(dest: Path, kept: set[str]) -> None:
    """Prune the store's consolidated metadata so dropped elements are not referenced.

    ``kept`` contains kept kind nodes (``"images"``) and kept element nodes (``"images/<name>"``).
    Handles the zarr-v3 root ``zarr.json`` form here; the v2 ``.zmetadata`` form is handled in Task 3.
    """
    zj = dest / "zarr.json"
    if zj.exists():
        doc = json.loads(zj.read_text())
        cm = doc.get("consolidated_metadata")
        if isinstance(cm, dict) and isinstance(cm.get("metadata"), dict):
            prefixes = tuple(e + "/" for e in kept if "/" in e)  # element nodes only; kind nodes would keep everything
            cm["metadata"] = {k: v for k, v in cm["metadata"].items() if k in kept or k.startswith(prefixes)}
            zj.write_text(json.dumps(doc, indent=2))


def _localize_to_cache(zarr_url: str, elements: list[str], dest_root: Path) -> Path:
    """Copy only ``elements`` (``"kind/name"``) + store metadata into a subset-keyed cache dir, pruned."""
    paths = sorted(set(elements))
    if not paths:
        raise ValueError("pass element paths in `elements`, or None for the whole store")
    stem = zarr_url.rstrip("/").rsplit("/", 1)[-1]
    stem = stem[:-5] if stem.endswith(".zarr") else stem
    subset = hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12]  # cache key, not security
    dest = dest_root / f"{stem}__{subset}.zarr"

    fs, root = fsspec.core.url_to_fs(zarr_url, **storage_options(zarr_url))
    if not fs.exists(root):
        raise FileNotFoundError(zarr_url)
    root = root.rstrip("/")
    kinds: dict[str, list[tuple[str, str]]] = {}
    for rel in paths:
        kind, _, name = rel.partition("/")
        if not name:
            raise ValueError(f"element path must be 'kind/name', got {rel!r}")
        kinds.setdefault(kind, []).append((name, rel))

    if dest.exists() and _localized_complete(dest, paths):
        return dest

    dest_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{stem}.partial-", dir=dest_root))
    try:
        for name in ("zarr.json", ".zattrs", ".zgroup", ".zmetadata", "zmetadata"):
            src = f"{root}/{name}"
            if fs.exists(src):
                fs.get(src, str(tmp / name))
        for kind, members in kinds.items():
            (tmp / kind).mkdir(exist_ok=True)
            for name in ("zarr.json", ".zattrs", ".zgroup"):
                src = f"{root}/{kind}/{name}"
                if fs.exists(src):
                    fs.get(src, str(tmp / kind / name))
            for name, rel in members:
                srcdir = f"{root}/{kind}/{name}"
                if not fs.exists(srcdir):
                    # `elements` is shadowed by the parameter; list siblings straight from the filesystem.
                    listed = (str(p).rstrip("/").rsplit("/", 1)[-1] for p in fs.ls(f"{root}/{kind}", detail=False))
                    present = sorted(
                        n for n in listed if not n.startswith(".") and not n.endswith((".json", ".zattrs", ".zgroup"))
                    )
                    raise FileNotFoundError(f"element {rel!r} not in store {zarr_url!r}; {kind} present: {present}")
                fs.get(srcdir + "/", str(tmp / kind / name) + "/", recursive=True)
        _prune_consolidated(tmp, set(paths) | set(kinds))
        _assert_localized_complete(tmp, paths)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(
            tmp, dest
        )  # replace is atomic, but the preceding rmtree(dest) + replace pair (stale dest only) is not
        return dest
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _localized_complete(dest: Path, paths: list[str]) -> bool:
    return all((dest / p).is_dir() and any((dest / p).iterdir()) for p in paths)


def _assert_localized_complete(dest: Path, paths: list[str]) -> None:
    missing = [p for p in paths if not ((dest / p).is_dir() and any((dest / p).iterdir()))]
    if missing:
        raise OSError(f"incomplete localized store at {dest}: missing {missing}")


def _describe(node: zarr.Array[Any] | zarr.Group, kind: str) -> dict[str, Any]:
    """Describe one zarr node from metadata only (no chunk reads); ``type`` is the element kind."""
    if isinstance(node, zarr.Array):
        return {"type": kind, "shape": tuple(node.shape), "dtype": str(node.dtype)}
    # multiscale image/label: full-resolution array is scale "s0" (zarr v3 format) or "0" (older)
    for key in ("s0", "0", "X"):
        if key in node:
            child = node[key]
            if isinstance(child, zarr.Array):
                return {"type": kind, "shape": tuple(child.shape), "dtype": str(child.dtype)}
            shape = child.attrs.get("shape")
            if shape is not None:  # sparse anndata X: dtype from the cheap X/data array metadata
                data = child["data"] if "data" in child else None
                dtype = str(data.dtype) if isinstance(data, zarr.Array) else None
                return {"type": kind, "shape": tuple(cast("list[int]", shape)), "dtype": dtype}
    # multiscale with non-standard scale keys: first array child (sorted)
    for _, child in sorted(node.members(), key=lambda kv: kv[0]):
        if isinstance(child, zarr.Array):
            return {"type": kind, "shape": tuple(child.shape), "dtype": str(child.dtype)}
    return {"type": kind, "shape": None, "dtype": None}


def elements(zarr_url: str) -> dict[str, dict[str, Any]]:
    """Introspect a SpatialData zarr without downloading array chunks.

    Parameters
    ----------
    zarr_url
        Store URL or local path.

    Returns
    -------
    Mapping ``"<kind>/<name>"`` to ``{"type", "shape", "dtype"}`` where ``type`` is the element kind (``images``/``labels``/``points``/``shapes``/``tables``); shape/dtype are read from zarr
    metadata only and may be ``None`` when not cheaply available.

    Raises
    ------
    FileNotFoundError
        If the store cannot be opened; the message names ``zarr_url``.
    """
    try:
        root = _open_group(zarr_url)
    except Exception as e:
        raise FileNotFoundError(f"Cannot open SpatialData zarr at {zarr_url!r}: {e}") from e
    out: dict[str, dict[str, Any]] = {}
    for kind in _ELEMENT_KINDS:
        if kind not in root:
            continue
        group = root[kind]
        if not isinstance(group, zarr.Group):
            continue
        for name, node in group.members():
            out[f"{kind}/{name}"] = _describe(node, kind)
    return out
