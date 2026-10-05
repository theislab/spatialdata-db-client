"""Cross-dataset gene search over the published gene index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from sddb import _catalog_schema as schema
from sddb._cache import fetch_catalog
from sddb.dataset import Results

if TYPE_CHECKING:
    from sddb.catalog import Catalog


def sibling_url(url: str, name: str) -> str:
    """Return ``url`` with its last path component replaced by ``name``."""
    return url.rsplit("/", 1)[0] + "/" + name if "/" in url else name


class GeneIndex:
    """Reader over ``gene_index.parquet``, bound to a Catalog for uid -> Dataset resolution."""

    def __init__(self, catalog: Catalog, url: str | None = None) -> None:
        self._catalog = catalog
        self.url = url or sibling_url(catalog.url, "gene_index.parquet")
        path = fetch_catalog(self.url, cache_dir=catalog._cache_dir)
        try:
            df = pd.read_parquet(path)
        except Exception as err:
            raise ValueError(f"gene index at {self.url} is missing or corrupt: {err}") from err
        for col, dtype in schema.GENE_INDEX_COLUMNS.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(pd.api.types.pandas_dtype(dtype))
                except (TypeError, ValueError):
                    pass  # left for validate() to report
        try:
            schema.validate(df, kind="gene_index")
        except schema.SchemaError as err:
            raise schema.SchemaError(f"gene index at {self.url} is invalid: {err}") from err
        self._df = df

    def datasets_with(
        self, symbol: str, *, min_fraction: float | None = None, validation: str | None = "pass"
    ) -> Results:
        """Catalog datasets whose gene index has ``symbol`` (case-insensitive).

        Parameters
        ----------
        symbol
            Gene symbol.
        min_fraction
            Keep only datasets where ``fraction_obs_detected >= min_fraction``.
        validation
            Keep datasets with this ``validation_status`` (default ``"pass"``, as in ``Catalog.query``); ``None`` disables.
        """
        df = self._df
        mask = df["symbol"].str.casefold() == symbol.casefold()
        if min_fraction is not None:
            mask &= df["fraction_obs_detected"] >= min_fraction
        uids = set(df.loc[mask.fillna(False).astype(bool), "uid"].dropna())
        cat = self._catalog.to_df()
        if validation is not None:
            cat = cat[cat["validation_status"] == validation]
        return Results(cat[cat["uid"].isin(uids)], source=self._catalog._source())
