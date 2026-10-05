"""Catalog fetch, validation, filtering and search."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from sddb import _catalog_schema as schema
from sddb._cache import fetch_catalog
from sddb.dataset import Results

DEFAULT_CATALOG_URL = "https://spatialdata-db.com/catalog.parquet"  # placeholder; finalized in WP5
_FUZZY_CUTOFF = 88


def _token_matches(token: str, value: str) -> bool:
    """Substring or close fuzzy match of a lowercase token against a facet value."""
    low = value.lower()
    if token == low or (len(token) >= 3 and token in low):
        return True
    return len(token) >= 4 and any(fuzz.ratio(token, w) >= _FUZZY_CUTOFF for w in re.split(r"[\s_\-/]+", low) if w)


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
            Catalog parquet URL or path; defaults to ``DEFAULT_CATALOG_URL``.
        cache_dir
            Cache directory override.
        version
            Dated snapshot; with the default url, selects ``catalog-<version>.parquet``.
        refresh
            Bypass the cache and re-download.
        """
        if url is None:
            url = DEFAULT_CATALOG_URL
            if version is not None:
                url = url.rsplit("/", 1)[0] + f"/catalog-{version}.parquet"
        self.url = url
        self.version = version
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

    @property
    def generated_at(self) -> str | None:
        """Generation timestamp from the parquet metadata, if present."""
        meta = pq.read_schema(self._path).metadata or {}
        value = meta.get(b"generated_at")
        return value.decode() if value else None

    def to_df(self) -> pd.DataFrame:
        """Return the full catalog as a DataFrame copy."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"<Catalog: {len(self)} datasets>"

    def query(
        self, *, validation: str | None = "pass", license_set: bool | None = None, **facets: str | list[str]
    ) -> Results:
        """Filter the catalog.

        Parameters
        ----------
        validation
            Keep rows with this ``validation_status`` (default ``"pass"``); ``None`` disables.
        license_set
            If True, keep only rows with a known license (``license_unknown`` is False).
        **facets
            Facet column -> value (equality) or list of values (isin).

        Raises
        ------
        ValueError
            If a keyword is not a facet column.
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
        for col, val in facets.items():
            mask &= df[col].isin(val if isinstance(val, list) else [val])
        return Results(df[mask.fillna(False).astype(bool)])

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
            return Results(self._df.iloc[0:0])
        mask = pd.Series(True, index=self._df.index)
        for col, hits in matched.items():
            mask &= self._df[col].isin(hits)
        if "validation_status" not in matched:
            mask &= self._df["validation_status"] == "pass"
        return Results(self._df[mask.fillna(False).astype(bool)], matched=matched)
