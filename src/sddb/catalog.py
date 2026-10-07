"""Catalog fetch, validation, filtering and search."""

from __future__ import annotations

import operator
import os
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from sddb import _catalog_schema as schema
from sddb._cache import _local_path, fetch_catalog
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Source

if TYPE_CHECKING:
    from sddb.genes import GeneIndex

DEFAULT_CATALOG_URL = "https://github.com/theislab/spatialdata-db-client/releases/latest/download/catalog.parquet"
_FUZZY_CUTOFF = 88
_RANGE_OPS = {"gte": operator.ge, "gt": operator.gt, "lte": operator.le, "lt": operator.lt}


def _split_facets(facets: Mapping[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str, Any]]]:
    """Separate plain equality facets from ``col__op`` range predicates (op in _RANGE_OPS)."""
    equality: dict[str, Any] = {}
    ranges: list[tuple[str, str, Any]] = []
    for key, val in facets.items():
        base, _, op = key.rpartition("__")
        if base and op in _RANGE_OPS:
            ranges.append((base, op, val))
        else:
            equality[key] = val
    return equality, ranges


def _token_matches(token: str, value: str) -> bool:
    """Match a lowercase token against a facet value: whole word first, fuzzy/substring for longer tokens."""
    low = value.lower()
    words = re.findall(r"\w+", low)
    if token == low or token in words:
        return True
    if len(token) >= 5 and token in low:
        return True
    return len(token) >= 4 and any(fuzz.ratio(token, w) >= _FUZZY_CUTOFF for w in words)


def facet_values(df: pd.DataFrame, field: str | None = None) -> list[str]:
    """Facet columns present in ``df``, or the sorted distinct non-null values of ``field``.

    Raises ``ValueError`` if ``field`` is not a known facet or is absent from ``df``.
    """
    if field is None:
        return [c for c in schema.FACETS if c in df.columns]
    if field not in schema.FACETS or field not in df.columns:
        raise ValueError(f"unknown facet {field!r}; valid facets: {list(schema.FACETS)}")
    return sorted(str(v) for v in df[field].dropna().unique())


class Catalog:
    """The dataset catalog: fetched, cached and validated on construction."""

    def __init__(
        self,
        url: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        version: str | None = None,
        refresh: bool = False,
    ) -> None:
        """Fetch + cache + validate the catalog.

        Parameters
        ----------
        url
            Catalog parquet URL or path; falls back to the ``SDDB_CATALOG_URL`` env var, then ``DEFAULT_CATALOG_URL``.
        cache_dir
            Cache directory override.
        version
            Dated snapshot; only used when ``url`` is None (ignored if an explicit url is given), selects ``catalog-<version>.parquet``.
        refresh
            Bypass the cache and re-download.
        """
        if url is None:
            url = os.environ.get("SDDB_CATALOG_URL") or DEFAULT_CATALOG_URL
            if version is not None:
                url = url.rsplit("/", 1)[0] + f"/catalog-{version}.parquet"
        local = _local_path(url)
        if local is not None and not url.startswith("file://"):
            url = str(local.resolve())  # stable cache key regardless of cwd
        self.url = url
        self.version = version
        self._cache_dir = cache_dir
        self._refresh = refresh
        self._genes: GeneIndex | None = None
        self._path = fetch_catalog(url, cache_dir=cache_dir, refresh=refresh)
        try:
            df = pd.read_parquet(self._path)
        except Exception as err:
            raise ValueError(f"catalog at {url} is missing or corrupt: {err}") from err
        for col, dtype in schema.CATALOG_COLUMNS.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(pd.api.types.pandas_dtype(dtype))
                except (TypeError, ValueError):
                    pass  # left for validate() to report
        try:
            schema.validate(df)
        except schema.SchemaError as err:
            raise schema.SchemaError(f"catalog at {url} is invalid: {err}") from err
        self._df = df

    @classmethod
    def from_file(cls, path: str | Path, *, cache_dir: str | Path | None = None) -> Catalog:
        """Load a catalog from a local ``catalog.parquet`` (absolute or relative path, or ``file://`` URL).

        Parameters
        ----------
        path
            Local parquet path; validated against the catalog schema like any other catalog.
        cache_dir
            Cache directory override.
        """
        return cls(str(path), cache_dir=cache_dir)

    @property
    def generated_at(self) -> str | None:
        """Generation timestamp from the parquet metadata, if present."""
        meta = pq.read_metadata(self._path).metadata or {}
        value = meta.get(b"sddb_generated_at")
        return value.decode() if value else None

    def _source(self) -> Source:
        return Source(self.url, self._cache_dir, self.version, self._refresh)

    @property
    def genes(self) -> GeneIndex:
        """Cross-dataset gene index bound to this catalog (loaded lazily)."""
        if self._genes is None:
            from sddb.genes import GeneIndex

            self._genes = GeneIndex(self)
        return self._genes

    def to_df(self) -> pd.DataFrame:
        """Return the full catalog as a DataFrame copy."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"<Catalog: {len(self)} datasets>"

    def query(
        self,
        *,
        validation: str | None = "pass",
        license_set: bool | None = None,
        noncommercial: bool | None = None,
        search: str | None = None,
        expressing: str | None = None,
        min_fraction: float | None = None,
        **facets: str | float | list[str] | tuple[str, ...] | set[str],
    ) -> SpatialDataCohort:
        """Filter the catalog.

        Parameters
        ----------
        validation
            Keep rows with this ``validation_status`` (default ``"pass"``); ``None`` disables.
        license_set
            If True, keep only rows with a known license (``license_unknown`` is False); False behaves like None.
        noncommercial
            If False, keep only commercial-ok data (``license_noncommercial`` is False) — the common
            "exclude NonCommercial" filter; if True, keep only NonCommercial data; ``None`` disables.
        search
            Free text, as in :meth:`search`; the result is intersected with the other filters and
            carries the ``matched`` facet values. With ``validation=None`` (or ``"all"``), ``search``
            still imposes its own ``validation_status == "pass"`` filter unless ``validation_status``
            was itself matched, so failed rows do not survive a ``search``.
        expressing
            Gene symbol; keep datasets where it is expressed (see ``genes.ranked``). The first use
            downloads the gene index (~142 MB, cached).
        min_fraction
            Minimum fraction of expressing cells for ``expressing``; requires ``expressing``.
        **facets
            Facet column -> value (equality) or list/tuple/set of values (isin). ``license`` and
            ``license_noncommercial`` are facets, so you can also filter by exact license id.
            ``<col>__gte`` / ``__gt`` / ``__lte`` / ``__lt`` apply a numeric range on a column in
            ``RANGE_COLUMNS`` (e.g. ``n_obs__gte=50_000``); missing values never match.

        Raises
        ------
        ValueError
            If a keyword is not a facet or range column, a facet column is absent from the loaded catalog,
            a range value is not numeric, or ``min_fraction`` is given without ``expressing``.
        """
        if min_fraction is not None and expressing is None:
            raise ValueError("min_fraction requires expressing=<symbol>")
        equality, ranges = _split_facets(facets)
        bad = [k for k in equality if k not in schema.FACETS]
        if bad:
            raise ValueError(f"unknown facet(s) {bad}; valid facets: {list(schema.FACETS)}")
        df = self._df
        unknown_range = [c for c, _, _ in ranges if c not in schema.RANGE_COLUMNS]
        if unknown_range:
            raise ValueError(f"unknown range column(s) {unknown_range}; valid: {list(schema.RANGE_COLUMNS)}")
        absent_range = [c for c, _, _ in ranges if c not in df.columns]
        if absent_range:
            raise ValueError(f"range column(s) {absent_range} not present in this catalog")
        mask = pd.Series(True, index=df.index)
        if validation is not None:
            mask &= df["validation_status"] == validation
        if license_set and "license_unknown" in df.columns:
            mask &= df["license_unknown"].fillna(True) == False  # noqa: E712
        if noncommercial is not None and "license_noncommercial" in df.columns:
            mask &= df["license_noncommercial"].fillna(False) == noncommercial
        for col, val in equality.items():
            if col not in df.columns:
                raise ValueError(f"facet column {col!r} is not present in this catalog (columns: {list(df.columns)})")
            mask &= df[col].isin(list(val) if isinstance(val, (list, tuple, set)) else [val])
        for col, op, val in ranges:
            if isinstance(val, (list, tuple, set)):
                raise ValueError(f"range value for {col}__{op} must be a scalar number, got {val!r}")
            try:
                mask &= _RANGE_OPS[op](df[col], val)
            except TypeError as err:
                raise ValueError(f"range value for {col}__{op} must be numeric, got {val!r}") from err
        keep = mask.fillna(False).astype(bool)
        # search and expressing each constrain the surviving uids; intersect into one mask, build SpatialDataCohort once.
        matched: dict[str, list[str]] | None = None
        if search is not None and keep.any():
            hits = self.search(search)
            matched = hits.matched or None
            keep &= df["uid"].isin(hits.to_df()["uid"])
        if expressing is not None and keep.any():  # skip the ~142 MB gene index when nothing is left to filter
            ranked = self.genes.ranked(expressing, min_fraction=min_fraction, validation=validation)
            keep &= df["uid"].isin(ranked["uid"])
        return SpatialDataCohort(df[keep], matched=matched, source=self._source())

    def search(self, text: str) -> SpatialDataCohort:
        """Deterministic text search over facet values.

        Tokens are fuzzy/substring-matched against the distinct values of each facet; matched values
        within a facet are OR-ed and facets are AND-ed. Rows that failed validation are excluded unless
        ``validation_status`` itself was matched. Nothing matching returns an empty SpatialDataCohort with a warning.
        """
        tokens = re.findall(r"\w+", text.lower())
        matched: dict[str, list[str]] = {}
        for col in schema.FACETS:
            if col not in self._df.columns:
                continue
            values = sorted(str(v) for v in self._df[col].dropna().unique())
            hits = [v for v in values if any(_token_matches(t, v) for t in tokens)]
            if hits:
                matched[col] = hits
        if not matched:
            warnings.warn(f"nothing matched {text!r}", stacklevel=2)
            return SpatialDataCohort(self._df.iloc[0:0], source=self._source())
        mask = pd.Series(True, index=self._df.index)
        for col, hits in matched.items():
            mask &= self._df[col].isin(hits)
        if "validation_status" not in matched:
            mask &= self._df["validation_status"] == "pass"
        return SpatialDataCohort(self._df[mask.fillna(False).astype(bool)], matched=matched, source=self._source())
