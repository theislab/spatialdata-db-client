"""Dataset records and result sets resolved from the catalog."""

from __future__ import annotations

import webbrowser
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import pandas as pd

from sddb import remote

if TYPE_CHECKING:
    from spatialdata import SpatialData

    from sddb.manifest import Manifest

_VITESSCE = "https://vitessce.io/?url="


@dataclass(frozen=True)
class Source:
    """Where a Results came from: the catalog url, cache dir and version (for sidecars/manifests)."""

    url: str | None = None
    cache_dir: str | Path | None = None
    version: str | None = None


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
        return remote.open_sdata(self.zarr_url, lazy=lazy)

    def elements(self) -> dict[str, dict[str, Any]]:
        """Return element shapes/dtypes without loading data."""
        return remote.elements(self.zarr_url)

    def viewer_url(self) -> str:
        """Build a vitessce.io viewer link from this row's ``vitessce_url`` (pure string transform).

        An ``s3://bucket/key`` config URL is rewritten to the LaminHub CORS proxy
        ``https://lamin.ai/storage/s3/bucket/key`` and wrapped as ``https://vitessce.io/?url=<proxied>``.

        Raises
        ------
        ValueError
            If the row has no ``vitessce_url``.
        """
        url: str | None = self.vitessce_url
        if not url:
            raise ValueError(f"dataset {self.uid} has no vitessce_url")
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
        return viz.render_config(remote.proxy_url(url))

    def __repr__(self) -> str:
        return f"<Dataset {self.uid} technology={self.technology} organism={self.organism}>"


class Results(Sequence[Dataset]):
    """An ordered, sliceable set of Dataset records from a query/search."""

    def __init__(
        self, df: pd.DataFrame, *, matched: dict[str, list[str]] | None = None, source: Source | None = None
    ) -> None:
        self._df = df.reset_index(drop=True)
        self.matched: dict[str, list[str]] = matched or {}
        self._source = source or Source()

    def citations(self, path: str | Path, *, bib_url: str | None = None) -> Path:
        """Write the BibTeX entries for the studies in this set to ``path``."""
        from sddb.citations import write_citations

        return write_citations(self, path, bib_url=bib_url)

    def download(
        self, dest: str | Path, *, workers: int = 4, pin_versions: bool = False, allow_version_change: bool = False
    ) -> Manifest:
        """Download every store under ``dest`` (resumable) and return the manifest.

        Raises ``ManifestVersionMismatch`` if ``dest`` holds a manifest pinned to another catalog version
        (unless ``allow_version_change``).
        """
        from sddb.manifest import download

        return download(
            self, dest, workers=workers, pin_versions=pin_versions, allow_version_change=allow_version_change
        )

    def to_df(self) -> pd.DataFrame:
        """Return the underlying rows as a DataFrame copy."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    @overload
    def __getitem__(self, i: int) -> Dataset: ...
    @overload
    def __getitem__(self, i: slice) -> Results: ...
    def __getitem__(self, i: int | slice) -> Dataset | Results:
        if isinstance(i, slice):
            return Results(self._df.iloc[i], matched=self.matched, source=self._source)
        return Dataset(self._df.iloc[i])

    def __iter__(self) -> Iterator[Dataset]:
        for i in range(len(self)):
            yield Dataset(self._df.iloc[i])

    def __repr__(self) -> str:
        return f"<Results: {len(self)} datasets>"
