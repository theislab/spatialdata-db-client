"""SpatialDataCohort: an ordered, sliceable set of Dataset records resolved from the catalog."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import pandas as pd

from sddb.dataset import Dataset, Source

if TYPE_CHECKING:
    from sddb.manifest import Manifest

_MODALITY_FLAGS = ("hne_image", "if_image", "ftu_annotation")  # catalog boolean columns


def group_counts(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Count rows per value of ``field``; missing values form an explicit ``"unknown"`` row."""
    if field not in df.columns:
        raise KeyError(f"no such field: {field!r}")
    key = df[field].astype("string").fillna("unknown")
    out = key.value_counts(dropna=False).rename_axis(field).reset_index(name="n")
    return out.sort_values("n", ascending=False, ignore_index=True)


def modalities(df: pd.DataFrame) -> pd.DataFrame:
    """Whether any row carries each modality flag (catalog columns only, no network)."""
    rows = [(flag, bool(df[flag].fillna(False).any())) for flag in _MODALITY_FLAGS if flag in df.columns]
    return pd.DataFrame(rows, columns=["modality", "present"])


def licenses(df: pd.DataFrame) -> pd.DataFrame:
    """Count rows per license."""
    return group_counts(df, "license") if "license" in df.columns else pd.DataFrame(columns=["license", "n"])


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column counts of present and unknown (missing) values."""
    rows = []
    for col in df.columns:
        present = int(df[col].notna().sum())
        rows.append((col, present, len(df) - present))
    return pd.DataFrame(rows, columns=["field", "present", "unknown"])


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers: object count, distinct studies, technologies."""
    return {
        "n_objects": len(df),
        "n_studies": int(df["study_id"].astype("string").fillna("unknown").nunique()) if "study_id" in df else None,
        "technologies": sorted(df["technology"].dropna().unique()) if "technology" in df else [],
    }


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

    def groupby(self, field: str) -> pd.DataFrame:
        """Count members per value of ``field`` (missing values bucketed as ``"unknown"``)."""
        return group_counts(self._df, field)

    def modalities(self) -> pd.DataFrame:
        """Which modalities are present, from catalog flag columns only (no network)."""
        return modalities(self._df)

    def licenses(self) -> pd.DataFrame:
        """Count members per license."""
        return licenses(self._df)

    def coverage(self) -> pd.DataFrame:
        """Per-column counts of present vs. unknown values."""
        return coverage(self._df)

    def summary(self) -> dict[str, Any]:
        """Headline numbers for this set."""
        return summary(self._df)

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
