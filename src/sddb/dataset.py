"""Dataset records and result sets resolved from the catalog."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, overload

import pandas as pd

from sddb import remote

if TYPE_CHECKING:
    from spatialdata import SpatialData

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
    return None if value is pd.NA or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)) else value


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
            Open in place (dask-backed) instead of copying the store to the cache.
        version
            Reserved; version pinning is not implemented yet.
        """
        if version is not None:
            raise NotImplementedError("version pinning is not implemented yet")
        return remote.open_sdata(self.zarr_url, lazy=lazy)

    def elements(self) -> dict[str, dict[str, Any]]:
        """Return element shapes/dtypes without loading data."""
        return remote.elements(self.zarr_url)

    def __repr__(self) -> str:
        return f"<Dataset {self.uid} technology={self.technology} organism={self.organism}>"


class Results(Sequence[Dataset]):
    """An ordered, sliceable set of Dataset records from a query/search."""

    def __init__(self, df: pd.DataFrame, *, matched: dict[str, list[str]] | None = None) -> None:
        self._df = df.reset_index(drop=True)
        self.matched: dict[str, list[str]] = matched or {}

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
            return Results(self._df.iloc[i], matched=self.matched)
        return Dataset(self._df.iloc[i])

    def __iter__(self) -> Iterator[Dataset]:
        for i in range(len(self)):
            yield Dataset(self._df.iloc[i])

    def __repr__(self) -> str:
        return f"<Results: {len(self)} datasets>"
