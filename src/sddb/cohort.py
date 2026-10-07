"""SpatialDataCohort: an ordered, sliceable set of Dataset records resolved from the catalog."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, overload

import pandas as pd

from sddb.dataset import Dataset, Source

if TYPE_CHECKING:
    from sddb.manifest import Manifest


class SpatialDataCohort(Sequence[Dataset]):
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

    def citations_text(self, *, bib_url: str | None = None) -> tuple[str, int]:
        """Return ``(BibTeX text, entry count)`` for the studies in this set, without writing a file."""
        from sddb.citations import citations_text

        return citations_text(self, bib_url=bib_url)

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
    def __getitem__(self, i: slice) -> SpatialDataCohort: ...
    def __getitem__(self, i: int | slice) -> Dataset | SpatialDataCohort:
        if isinstance(i, slice):
            return SpatialDataCohort(self._df.iloc[i], matched=self.matched, source=self._source)
        return Dataset(self._df.iloc[i])

    def __iter__(self) -> Iterator[Dataset]:
        for i in range(len(self)):
            yield Dataset(self._df.iloc[i])

    def __repr__(self) -> str:
        return f"<SpatialDataCohort: {len(self)} datasets>"
