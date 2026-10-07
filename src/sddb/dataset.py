"""Dataset records resolved from the catalog."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sddb import remote

if TYPE_CHECKING:
    from spatialdata import SpatialData

_VITESSCE = "https://vitessce.io/?url="


@dataclass(frozen=True)
class Source:
    """Where a SpatialDataCohort came from: the catalog url, cache dir and version (for sidecars/manifests)."""

    url: str | None = None
    cache_dir: str | Path | None = None
    version: str | None = None
    refresh: bool = False


_ATTRS = (
    "uid",
    "zarr_url",
    "technology",
    "assay",
    "organism",
    "tissue",
    "disease",
    "validation_status",
    "tier",
    "size_bytes",
)


def _clean(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return value  # container cell: present value, never scalar-isna
    return None if value is pd.NA or pd.isna(value) else value


class Dataset:
    """One catalog row: a published dataset. Column values are exposed as attributes."""

    def __init__(self, row: pd.Series) -> None:
        self._row = row

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._row.index:
            return _clean(self._row[name])
        if name in _ATTRS:
            return None
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def load(self, *, lazy: bool = True, version: str | None = None) -> SpatialData:
        """Open this dataset as SpatialData.

        Parameters
        ----------
        lazy
            Open a local store in place (dask-backed). Remote stores cannot be opened lazily yet
            (spatialdata upstream limitation); for a remote dataset pass ``lazy=False`` to copy the
            store to the cache and open it, or use :meth:`elements` to inspect without downloading.
        version
            Reserved; version pinning is not implemented yet.

        Raises
        ------
        NotImplementedError
            If the dataset is remote and ``lazy=True`` (the default).
        """
        if version is not None:
            raise NotImplementedError("version pinning is not implemented yet")
        try:
            return remote.open_sdata(self.zarr_url, lazy=lazy)
        except NotImplementedError as e:
            if "remote" in str(e).lower() and "lazy" in str(e).lower():
                raise NotImplementedError(
                    "Remote lazy open isn't supported upstream (spatialdata). "
                    "Download first: `SpatialDataCohort.download(dest)` then open the local store, "
                    "or use `load(lazy=False)` to materialise."
                ) from e
            raise

    def elements(self) -> dict[str, dict[str, Any]]:
        """Return element shapes/dtypes without loading data."""
        return remote.elements(self.zarr_url)

    def viewer_url(self) -> str:
        """Build a vitessce.io viewer link from this row's ``vitessce_url`` (pure string transform).

        If ``vitessce_url`` is already a vitessce.io viewer link (as in the published catalog) it is
        returned unchanged. Otherwise it is treated as a config-sidecar URL: an ``s3://bucket/key`` URL is rewritten to the LaminHub CORS proxy
        ``https://lamin.ai/storage/s3/bucket/key`` and wrapped as ``https://vitessce.io/?url=<proxied>``.

        Raises
        ------
        ValueError
            If the row has no ``vitessce_url``.
        """
        url: str | None = self.vitessce_url
        if not url:
            raise ValueError(f"dataset {self.uid} has no vitessce_url")
        if url.startswith("https://vitessce.io"):
            return url
        return _VITESSCE + remote.proxy_url(url)

    def view(self) -> None:
        """Open :meth:`viewer_url` in the default browser."""
        webbrowser.open(self.viewer_url())

    def view_interactive(self, *, mode: str = "config") -> Any:
        """In-notebook interactive view; requires the ``[viz]`` extra.

        ``mode="config"`` renders the published Vitessce config (``vitessce_url``); ``mode="sdata"``
        loads the store and renders it via easy_vitessce.
        """
        from sddb import viz

        if mode == "sdata":
            return viz.render_sdata(self.load())
        if mode != "config":
            raise ValueError(f"mode must be 'config' or 'sdata', got {mode!r}")
        url: str | None = self.vitessce_url
        if not url:
            raise ValueError(f"dataset {self.uid} has no vitessce_url")
        inlined = viz.config_from_vitessce_url(url)
        return viz.render_config(inlined if inlined is not None else remote.proxy_url(url))

    def __repr__(self) -> str:
        return f"<Dataset {self.uid} technology={self.technology} organism={self.organism}>"
