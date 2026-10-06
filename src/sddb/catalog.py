"""Catalog fetch, validation, filtering and search."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from sddb import _catalog_schema as schema
from sddb._cache import _local_path, fetch_catalog
from sddb.dataset import Results, Source

if TYPE_CHECKING:
    from sddb.genes import GeneIndex

DEFAULT_CATALOG_URL = "https://github.com/theislab/spatialdata-db-client/releases/latest/download/catalog.parquet"
_FUZZY_CUTOFF = 88


def _token_matches(token: str, value: str) -> bool:
    """Match a lowercase token against a facet value: whole word first, fuzzy/substring for longer tokens."""
    low = value.lower()
    words = re.findall(r"\w+", low)
    if token == low or token in words:
        return True
    if len(token) >= 5 and token in low:
        return True
    return len(token) >= 4 and any(fuzz.ratio(token, w) >= _FUZZY_CUTOFF for w in words)


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
        **facets: str | list[str] | tuple[str, ...] | set[str],
    ) -> Results:
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
        **facets
            Facet column -> value (equality) or list/tuple/set of values (isin). ``license`` and
            ``license_noncommercial`` are facets, so you can also filter by exact license id.

        Raises
        ------
        ValueError
            If a keyword is not a facet column, or a facet column is absent from the loaded catalog.
        """
        bad = [k for k in facets if k not in schema.FACETS]
        if bad:
            raise ValueError(f"unknown facet(s) {bad}; valid facets: {list(schema.FACETS)}")
        df = self._df
        mask = pd.Series(True, index=df.index)
        if validation is not None:
            mask &= df["validation_status"] == validation
        if license_set and "license_unknown" in df.columns:
            mask &= df["license_unknown"].fillna(True) == False  # noqa: E712
        if noncommercial is not None and "license_noncommercial" in df.columns:
            mask &= df["license_noncommercial"].fillna(False) == noncommercial
        for col, val in facets.items():
            if col not in df.columns:
                raise ValueError(f"facet column {col!r} is not present in this catalog (columns: {list(df.columns)})")
            mask &= df[col].isin(list(val) if isinstance(val, (list, tuple, set)) else [val])
        return Results(df[mask.fillna(False).astype(bool)], source=self._source())

    def search(self, text: str) -> Results:
        """Deterministic text search over facet values.

        Tokens are fuzzy/substring-matched against the distinct values of each facet; matched values
        within a facet are OR-ed and facets are AND-ed. Rows that failed validation are excluded unless
        ``validation_status`` itself was matched. Nothing matching returns an empty Results with a warning.
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
            return Results(self._df.iloc[0:0], source=self._source())
        mask = pd.Series(True, index=self._df.index)
        for col, hits in matched.items():
            mask &= self._df[col].isin(hits)
        if "validation_status" not in matched:
            mask &= self._df["validation_status"] == "pass"
        return Results(self._df[mask.fillna(False).astype(bool)], matched=matched, source=self._source())
