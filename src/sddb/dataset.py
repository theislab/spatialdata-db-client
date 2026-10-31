"""Dataset records resolved from the catalog."""

from __future__ import annotations

import re
import warnings
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import numpy as np
import pandas as pd

from sddb import remote
from sddb.roles import resolve as _resolve_roles

if TYPE_CHECKING:
    from spatialdata import SpatialData

_VITESSCE = "https://vitessce.io/?url="
_STORE_ID = re.compile(r"\.lamindb/([A-Za-z0-9]+)\.zarr")


def _loads_other_store(vitessce_url: str, zarr_url: str | None) -> bool:
    """True when a viewer link loads a ``.lamindb/<id>.zarr`` store other than the one in ``zarr_url``.

    Published links embed the Vitessce config URL-encoded twice, so the link is decoded twice before the
    store ids are read. A link or ``zarr_url`` without a store id is not checked.
    """
    linked = set(_STORE_ID.findall(unquote(unquote(vitessce_url))))
    own = set(_STORE_ID.findall(zarr_url or ""))
    return bool(linked and own) and linked != own


def usable_vitessce_url(uid: Any, vitessce_url: Any, zarr_url: Any) -> str | None:
    """``vitessce_url``, or None when it is missing or loads another dataset's store (then with a warning).

    Every path that returns or renders a ``vitessce_url`` goes through this check, as the website does.
    """
    url = _clean(vitessce_url)
    if not url:
        return None
    if _loads_other_store(str(url), _clean(zarr_url)):
        warnings.warn(
            f"the viewer link for {uid} loads another dataset's store; it is hidden until the catalog is fixed",
            UserWarning,
            stacklevel=3,
        )
        return None
    return str(url)


@dataclass(frozen=True)
class Source:
    """Where a SpatialDataCohort came from: the catalog url, cache dir, version and file hash (for sidecars/manifests).

    ``sha256`` is ``"sha256:<hex>"`` of the catalog parquet the cohort was built from, or None if unknown.
    """

    url: str | None = None
    cache_dir: str | Path | None = None
    version: str | None = None
    refresh: bool = False
    sha256: str | None = None


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
        (``"kind/name"``) copies only those elements. Both together copy only those elements into
        ``dest/<uid>.zarr`` and record them in ``dest/manifest.json`` as a ``"partial"`` entry; a later
        ``download(dest)`` fetches the whole store. If ``dest`` already holds the complete store, ``elements``
        opens it whole.
        """
        from sddb.cohort import SpatialDataCohort

        return SpatialDataCohort(self._row.to_frame().T).to_list(elements=elements, dest=dest, workers=1)[0]

    def elements(self) -> dict[str, dict[str, Any]]:
        """Return element shapes/dtypes without loading data."""
        return remote.elements(self.zarr_url)

    def resolve(self, roles: Mapping[str, str]) -> dict[str, str]:
        """Resolve role regexes to this store's element paths (``"kind/name"``); see :func:`sddb.resolve`."""
        return _resolve_roles(self.elements().keys(), roles)

    def viewer_url(self) -> str | None:
        """Build a vitessce.io viewer link from this row's ``vitessce_url`` (pure string transform).

        If ``vitessce_url`` is already a vitessce.io viewer link (as in the published catalog) it is
        returned unchanged. Otherwise it is treated as a config-sidecar URL: an ``s3://bucket/key`` URL is
        rewritten to the LaminHub CORS proxy ``https://lamin.ai/storage/s3/bucket/key`` and wrapped as
        ``https://vitessce.io/?url=<proxied>``.

        Some published links load another dataset's store (333 of 6087 rows in the 2026-10-28 catalog). The
        store id in the link (``.lamindb/<id>.zarr``) is compared with the one in ``zarr_url``, and on a
        mismatch the link is hidden: this returns ``None`` with a warning, as the website does.

        Warns
        -----
        UserWarning
            If the link loads another dataset's store.

        Raises
        ------
        ValueError
            If the row has no ``vitessce_url``.
        """
        if not self.vitessce_url:
            raise ValueError(f"dataset {self.uid} has no vitessce_url")
        url = usable_vitessce_url(self.uid, self.vitessce_url, self.zarr_url)
        if url is None:
            return None
        if url.startswith("https://vitessce.io"):
            return url
        return _VITESSCE + remote.proxy_url(url)

    def view(self) -> None:
        """Open :meth:`viewer_url` in the default browser.

        Raises
        ------
        ValueError
            If the row has no ``vitessce_url``, or its link loads another dataset's store (with the warning
            from :meth:`viewer_url`).
        """
        url = self.viewer_url()
        if url is None:
            raise ValueError(f"dataset {self.uid} has no usable viewer link")
        webbrowser.open(url)

    def view_interactive(self, *, mode: str = "config") -> Any:
        """In-notebook interactive view; requires the ``[viz]`` extra.

        ``mode="config"`` renders the published Vitessce config (``vitessce_url``); ``mode="sdata"``
        loads the store and renders it via easy_vitessce.

        Raises
        ------
        ValueError
            With ``mode="config"``, if the row has no ``vitessce_url`` or its link loads another dataset's
            store (with the warning from :meth:`viewer_url`).
        """
        from sddb import viz

        if mode == "sdata":
            return viz.render_sdata(self.to_sdata())
        if mode != "config":
            raise ValueError(f"mode must be 'config' or 'sdata', got {mode!r}")
        url = usable_vitessce_url(self.uid, self.vitessce_url, self.zarr_url)
        if url is None:
            raise ValueError(f"dataset {self.uid} has no usable vitessce_url")
        inlined = viz.config_from_vitessce_url(url)
        return viz.render_config(inlined if inlined is not None else remote.proxy_url(url))

    def __repr__(self) -> str:
        return f"<Dataset {self.uid} technology={self.technology} organism={self.organism}>"
