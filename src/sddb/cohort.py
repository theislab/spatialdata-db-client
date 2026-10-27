"""SpatialDataCohort: an ordered, sliceable set of Dataset records resolved from the catalog."""

from __future__ import annotations

import os
import shutil
import threading
import warnings
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import pandas as pd

from sddb._cache import cache_dir
from sddb.dataset import Dataset, Source
from sddb.remote import _localize_to_cache, _open_fs

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

    from sddb.adapter import TableAdapter
    from sddb.catalog import Catalog
    from sddb.manifest import CohortManifest, Manifest

_MODALITY_FLAGS = ("hne_image", "if_image", "ftu_annotation")  # catalog boolean columns


def group_counts(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Count rows per value of ``field``; missing values form an explicit ``"unknown"`` row."""
    if field not in df.columns:
        raise KeyError(f"no such field: {field!r}")
    col = df[field]
    if any(not isinstance(v, str) and hasattr(v, "__iter__") for v in col.dropna()):  # list-valued facet
        col = col.map(lambda v: list(v) if v is not None and hasattr(v, "__iter__") and not isinstance(v, str) else v)
        col = col.map(lambda v: v if isinstance(v, list) and v else [None]).explode()
    key = col.astype("string").fillna("unknown")
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


def _cached_store(uid: str, url: str, root: Path) -> Path:
    """``root/<uid>.zarr``, copied once via a ``.part`` dir renamed on success, so a present store is complete."""
    from sddb.manifest import _copy_store

    final = root / f"{uid}.zarr"
    if final.is_dir():
        return final
    root.mkdir(parents=True, exist_ok=True)
    part = root / f"{uid}.zarr.part-{os.getpid()}-{threading.get_ident()}"
    shutil.rmtree(part, ignore_errors=True)
    try:
        _copy_store(url, part)
        part.rename(final)
    except OSError:
        shutil.rmtree(part, ignore_errors=True)
        if not final.is_dir():  # a concurrent caller finishing first is fine; anything else is an error
            raise
    return final


class SpatialDataCohort(Sequence[Dataset]):
    """An ordered, sliceable set of Dataset records from a query/search."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        matched: dict[str, list[str]] | None = None,
        source: Source | None = None,
        filter: dict[str, Any] | None = None,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self.matched: dict[str, list[str]] = matched or {}
        self._source = source or Source()
        self._filter: dict[str, Any] = filter or {}
        self.split_manifest: Any = None  # set when this cohort is a fold of a split()

    def freeze(self) -> CohortManifest:
        """Pin this selection to its exact members and a deterministic hash. Offline/cheap.

        The recorded filter captures effective defaults (e.g. ``validation="pass"``). Cohorts built
        by paths other than ``query``/``search`` may carry an empty filter.
        """
        from datetime import UTC, datetime

        from sddb.manifest import SCHEMA_VERSION, CohortManifest, cohort_hash

        flt = dict(self._filter)
        return CohortManifest(
            SCHEMA_VERSION,
            datetime.now(UTC).isoformat(),
            self._source,
            flt,
            cohort_hash(flt, self._df["uid"]),
            self._df.copy(),
        )

    @classmethod
    def from_manifest(cls, m: CohortManifest, *, catalog: Catalog | None = None) -> SpatialDataCohort:
        """Rebuild a cohort pinned to exactly the manifest's members (offline — no catalog re-query).

        If ``catalog`` is given, warn once about any members a newer version supersedes (see
        :meth:`check_updates`).
        """
        coh = cls(m.members, source=m.source, filter=m.filter)
        if catalog is not None:
            coh.check_updates(catalog, warn=True)
        return coh

    def verify(self, *, workers: int = 8) -> None:
        """Liveness check: raise ``CohortDriftError`` if any pinned store no longer resolves.

        A pinned version may stop resolving when a dataset is republished; this detects that. Not called by
        :meth:`download`, which runs its own inline liveness loop; not run by cheap ops.
        """
        from sddb.manifest import CohortDriftError

        pairs = [(str(r.uid), str(r.zarr_url)) for r in self._df.itertuples(index=False)]

        def resolves(uid: str, url: str) -> tuple[str, bool]:
            try:
                fs, root = _open_fs(url)
                return uid, bool(fs.exists(root))
            except Exception:
                return uid, False

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(lambda p: resolves(*p), pairs))
        missing = {uid: "store does not resolve" for uid, ok in results if not ok}
        if missing:
            raise CohortDriftError(missing)

    def check_updates(self, catalog: Catalog, *, warn: bool = True) -> dict[str, str]:
        """Return ``{pinned_uid: current_uid}`` for members a newer catalog version supersedes.

        Matches each pinned ``uid[:16]`` (stem) against the catalog's current uid for that stem (the
        catalog is latest-only). Warns once listing the superseded members when ``warn``.
        """
        current = catalog.to_df()
        latest_by_stem = {str(u)[:16]: str(u) for u in current["uid"]}
        superseded: dict[str, str] = {}
        for u in self._df["uid"]:
            u = str(u)
            cur = latest_by_stem.get(u[:16])
            if cur is not None and cur != u:
                superseded[u] = cur
        if superseded and warn:
            warnings.warn(
                f"{len(superseded)} member(s) superseded by a newer version: {sorted(superseded)}",
                stacklevel=2,
            )
        return superseded

    def _derive(self, df: pd.DataFrame, filter: dict[str, Any]) -> SpatialDataCohort:
        """A new cohort over ``df`` carrying this cohort's source and split manifest (used by subsetting/splits)."""
        sub = SpatialDataCohort(df, matched=self.matched, source=self._source, filter=filter)
        sub.split_manifest = self.split_manifest
        return sub

    def split(
        self, *, by: str = "study_id", train: float, val: float, test: float, seed: int
    ) -> dict[str, SpatialDataCohort]:
        """Group-safe deterministic split into ``{fold: SpatialDataCohort}``.

        Each returned cohort carries the derived ``SplitManifest`` as ``.split_manifest`` (pinned to
        this cohort's hash). Splitting is always group-safe (``mode="group"``).
        """
        from sddb.manifest import cohort_hash
        from sddb.split import make_split

        parent = cohort_hash(self._filter, self._df["uid"])
        sm = make_split(self._df, parent, by=by, train=train, val=val, test=test, seed=seed, mode="group")
        uid_str = self._df["uid"].astype("string")
        folds: dict[str, SpatialDataCohort] = {}
        for fold in ("train", "val", "test"):
            want = [u for u, f in sm.assignments.items() if f == fold]
            sub = self._derive(self._df[uid_str.isin(want)], self._filter)
            sub.split_manifest = sm
            folds[fold] = sub
        return folds

    @property
    def members(self) -> pd.DataFrame:
        """The underlying catalog rows (a view; treat as read-only — use methods to derive new cohorts)."""
        return self._df

    def filter(self, **facets: Any) -> SpatialDataCohort:
        """Return a new cohort keeping rows matching every ``column=value`` (a list/tuple/set means isin)."""
        df = self._df
        mask = pd.Series(True, index=df.index)
        norm: dict[str, Any] = {}
        for col, val in facets.items():
            if col not in df.columns:
                raise ValueError(f"unknown column {col!r}; present: {list(df.columns)}")
            mask &= df[col].isin(list(val) if isinstance(val, (list, tuple, set)) else [val])
            norm[col] = sorted(val, key=str) if isinstance(val, (list, tuple, set)) else val
        return self._derive(df[mask.fillna(False).astype(bool)], {**self._filter, **norm})

    def where(self, mask: pd.Series) -> SpatialDataCohort:
        """Return a new cohort keeping rows where the boolean ``mask`` (aligned to ``members``) is True."""
        return self._derive(self._df[mask.fillna(False).astype(bool)], self._filter)

    def __or__(self, other: object) -> SpatialDataCohort:
        """Union, deduped by uid (first occurrence wins). Provenance filter is dropped (no longer a pure facet set)."""
        if not isinstance(other, SpatialDataCohort):
            return NotImplemented
        df = pd.concat([self._df, other._df]).drop_duplicates("uid", keep="first")
        out = self._derive(df, {})
        if other.split_manifest is not self.split_manifest:
            out.split_manifest = None  # mixes splits (or split + non-split): no single manifest applies
        return out

    def __sub__(self, other: object) -> SpatialDataCohort:
        """Difference: drop any member whose uid is in ``other``."""
        if not isinstance(other, SpatialDataCohort):
            return NotImplemented
        return self._derive(self._df[~self._df["uid"].isin(other._df["uid"])], self._filter)

    def citations(self, path: str | Path, *, bib_url: str | None = None) -> Path:
        """Write the BibTeX entries for the studies in this set to ``path``."""
        from sddb.citations import write_citations

        return write_citations(self, path, bib_url=bib_url)

    def citations_text(self, *, bib_url: str | None = None) -> tuple[str, int]:
        """Return ``(BibTeX text, entry count)`` for the studies in this set, without writing a file."""
        from sddb.citations import citations_text

        return citations_text(self, bib_url=bib_url)

    def download(
        self,
        dest: str | Path,
        *,
        workers: int = 4,
        pin_versions: bool = False,
        allow_version_change: bool = False,
        strict: bool = False,
    ) -> Manifest:
        """Download every store under ``dest`` (resumable) and return the manifest.

        Raises ``ManifestVersionMismatch`` if ``dest`` holds a manifest pinned to another catalog version
        (unless ``allow_version_change``). A missing pinned store falls back to the latest version with a
        loud warning unless ``strict=True`` (then it raises ``CohortDriftError``).
        """
        from sddb.manifest import download

        return download(
            self,
            dest,
            workers=workers,
            pin_versions=pin_versions,
            allow_version_change=allow_version_change,
            strict=strict,
        )

    def to_anndata(
        self,
        *,
        table: str | None = None,
        obs_columns: Sequence[str] = (),
        elements: Sequence[str] | None = None,
        adapter: TableAdapter | None = None,
        max_workers: int = 8,
    ) -> AnnData:
        """Concatenate each member's table into one AnnData (outer join, obs[uid]+obs[study_id]).

        See :func:`sddb.concat.concat_tables` for ``obs_columns``, ``elements``, ``adapter`` and ``max_workers``.
        """
        from sddb.concat import concat_tables

        return concat_tables(
            self._df,
            table=table,
            obs_columns=obs_columns,
            elements=elements,
            adapter=adapter,
            max_workers=max_workers,
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

    def to_list(
        self, *, elements: Sequence[str] | None = None, dest: str | Path | None = None, workers: int = 4
    ) -> list[SpatialData]:
        """Open every member as a SpatialData, in member order.

        Stores are fetched in parallel and reused on later calls: into the sddb cache (keyed by full uid,
        so a republished version is fetched afresh), or with ``dest`` into ``dest/<uid>.zarr`` plus a
        manifest, exactly as :meth:`download`. ``elements`` (``"kind/name"``) copies only those elements
        (cache only).
        """
        from spatialdata import read_zarr

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(read_zarr, self._local_stores(elements, dest, workers)))

    def to_dict(
        self,
        *,
        key: str = "uid",
        elements: Sequence[str] | None = None,
        dest: str | Path | None = None,
        workers: int = 4,
    ) -> dict[str, SpatialData]:
        """Like :meth:`to_list`, keyed by the ``key`` column (default the full uid), which must be unique and set."""
        if key not in self._df.columns:
            raise KeyError(f"no such column: {key!r}")
        keys = self._df[key]
        if keys.isna().any() or keys.duplicated().any():
            raise ValueError(f"column {key!r} is missing or repeats values in this cohort; pick a unique key")
        return dict(zip(keys.astype(str), self.to_list(elements=elements, dest=dest, workers=workers), strict=True))

    def _local_stores(self, elements: Sequence[str] | None, dest: str | Path | None, workers: int) -> list[Path]:
        urls = [str(u) for u in self._df["zarr_url"]]
        if elements is not None:
            if dest is not None:
                raise ValueError("elements= copies into the cache; it cannot be combined with dest=")
            root = cache_dir()
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                return list(pool.map(lambda u: _localize_to_cache(u, list(elements), root), urls))
        if dest is not None:
            failed = [e.uid for e in self.download(dest, workers=workers).entries if e.status != "complete"]
            if failed:
                raise RuntimeError(f"download failed for {failed}; see {Path(dest) / 'manifest.json'}")
            return [Path(dest) / f"{uid}.zarr" for uid in self._df["uid"]]
        root = cache_dir() / "stores"
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(
                pool.map(lambda p: _cached_store(*p, root), zip(self._df["uid"].astype(str), urls, strict=False))
            )

    def __len__(self) -> int:
        return len(self._df)

    @overload
    def __getitem__(self, i: int) -> Dataset: ...
    @overload
    def __getitem__(self, i: slice) -> SpatialDataCohort: ...
    def __getitem__(self, i: int | slice) -> Dataset | SpatialDataCohort:
        if isinstance(i, slice):
            return self._derive(self._df.iloc[i], self._filter)
        return Dataset(self._df.iloc[i])

    def __iter__(self) -> Iterator[Dataset]:
        for i in range(len(self)):
            yield Dataset(self._df.iloc[i])

    def __repr__(self) -> str:
        return f"<SpatialDataCohort: {len(self)} datasets>"
