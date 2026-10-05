"""Anonymous remote read primitives for SpatialData zarr stores."""

from __future__ import annotations

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


def open_sdata(zarr_url: str, *, lazy: bool = True, cache_dir: str | Path | None = None) -> SpatialData:
    """Open a SpatialData zarr from a URL or local path, anonymously.

    Parameters
    ----------
    zarr_url
        Store URL or local path.
    lazy
        If True, open in place (dask-backed, no bulk download). If False, copy the store into
        the cache directory first and open the local copy.
    cache_dir
        Cache directory override (only used when ``lazy=False``).

    Returns
    -------
    The opened SpatialData object.

    Raises
    ------
    FileNotFoundError
        If the store cannot be opened; the message names ``zarr_url``.
    """
    from spatialdata import read_zarr
    from upath import UPath

    try:
        if not lazy:
            local = _copy_to_cache(zarr_url, _cache_dir(cache_dir))
            return read_zarr(local)
        if "://" in zarr_url:
            # NOTE: spatialdata 0.8.0 cannot read remote stores at all (see tests/test_remote.py xfail)
            return read_zarr(UPath(zarr_url, **storage_options(zarr_url)))
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


def _describe(node: zarr.Array[Any] | zarr.Group) -> dict[str, Any]:
    """Describe one zarr node from metadata only (no chunk reads)."""
    if isinstance(node, zarr.Array):
        return {"type": "array", "shape": tuple(node.shape), "dtype": str(node.dtype)}
    # multiscale image/label: full-resolution array is scale "s0" (zarr v3 format) or "0" (older)
    for key in ("s0", "0", "X"):
        if key in node:
            child = node[key]
            if isinstance(child, zarr.Array):
                return {"type": "group", "shape": tuple(child.shape), "dtype": str(child.dtype)}
            shape = child.attrs.get("shape")
            if shape is not None:  # sparse anndata X
                return {"type": "group", "shape": tuple(cast("list[int]", shape)), "dtype": None}
    return {"type": "group", "shape": None, "dtype": None}


def elements(zarr_url: str) -> dict[str, dict[str, Any]]:
    """Introspect a SpatialData zarr without downloading array chunks.

    Parameters
    ----------
    zarr_url
        Store URL or local path.

    Returns
    -------
    Mapping ``"<kind>/<name>"`` to ``{"type", "shape", "dtype"}``; shape/dtype are read from zarr
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
            out[f"{kind}/{name}"] = _describe(node)
    return out
