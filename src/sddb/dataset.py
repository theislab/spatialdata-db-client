"""Dataset records resolved from the catalog."""

from __future__ import annotations

import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sddb import remote
from sddb.roles import resolve as _resolve_roles

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

    def to_sdata(self, *, elements: list[str] | None = None, dest: str | Path | None = None) -> SpatialData:
        """Open this dataset as SpatialData, fetched and cached like :meth:`SpatialDataCohort.to_list`.

        A remote store is copied once into the cache (keyed by full uid) and reused; a local store opens in
        place. ``dest`` keeps the store in ``dest/<uid>.zarr`` (the ``download()`` layout); ``elements``
        (``"kind/name"``) copies only those elements.
        """
        from sddb.cohort import SpatialDataCohort

        return SpatialDataCohort(self._row.to_frame().T).to_list(elements=elements, dest=dest, workers=1)[0]

    def elements(self) -> dict[str, dict[str, Any]]:
        """Return element shapes/dtypes without loading data."""
        return remote.elements(self.zarr_url)

    def resolve(self, roles: Mapping[str, str]) -> dict[str, str]:
        """Resolve role regexes to this store's element paths (``"kind/name"``); see :func:`sddb.resolve`."""
        return _resolve_roles(self.elements().keys(), roles)

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
            return viz.render_sdata(self.to_sdata())
        if mode != "config":
            raise ValueError(f"mode must be 'config' or 'sdata', got {mode!r}")
        url: str | None = self.vitessce_url
        if not url:
            raise ValueError(f"dataset {self.uid} has no vitessce_url")
        inlined = viz.config_from_vitessce_url(url)
        return viz.render_config(inlined if inlined is not None else remote.proxy_url(url))

    def __repr__(self) -> str:
        return f"<Dataset {self.uid} technology={self.technology} organism={self.organism}>"
