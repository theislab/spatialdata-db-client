"""Cross-dataset gene search over the published gene index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from sddb import _catalog_schema as schema
from sddb._cache import fetch_catalog
from sddb.cohort import SpatialDataCohort

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
        path = fetch_catalog(self.url, cache_dir=catalog._cache_dir, refresh=catalog._refresh)
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
        # Casefold the token columns once (the index is ~11M rows) so resolve() is a cheap per-token mask.
        self._sym = df["symbol"].str.casefold()
        self._fid = df["feature_id"].str.casefold()  # may hold pd.NA (legacy rows): compare vectorised only

    def resolve(self, tokens: str | list[str], *, min_fraction: float | None = None) -> dict[str, set[str]]:
        """Map each token to the uids of datasets expressing it.

        A token (casefolded, stripped) matches a gene-index row by symbol OR Ensembl ``feature_id``;
        an ambiguous symbol therefore unions all its feature ids. Unmatched tokens map to an empty set.
        Keys keep the original token spelling.
        """
        df = self._df
        base = (
            (df["fraction_obs_detected"] >= min_fraction).fillna(False).astype(bool)
            if min_fraction is not None
            else None
        )
        sym, fid = self._sym, self._fid
        out: dict[str, set[str]] = {}
        for tok in [tokens] if isinstance(tokens, str) else tokens:
            key = tok.strip().casefold()
            mask = ((sym == key) | (fid == key)).fillna(False).astype(bool)
            if base is not None:
                mask &= base
            out[tok] = {str(u) for u in df.loc[mask, "uid"].dropna()}
        return out

    def where_expressed(
        self,
        genes: str | list[str],
        *,
        mode: str = "all",
        min_fraction: float | None = None,
        validation: str | None = None,
    ) -> SpatialDataCohort:
        """Catalog datasets expressing ``genes`` (symbols or Ensembl ids, case-insensitive), as a cohort.

        Parameters
        ----------
        genes
            One gene or a list; each may be a symbol or an Ensembl feature id.
        mode
            ``"all"`` (default) keeps datasets expressing every gene (AND); ``"any"`` the union.
        min_fraction
            Per gene, keep only datasets where ``fraction_obs_detected >= min_fraction``.
        validation
            Keep datasets with this ``validation_status``; the default ``None`` includes bronze/non-pass
            datasets (flagged via the cohort's ``validation_status``/``tier`` columns).
        """
        if mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', got {mode!r}")
        sets = list(self.resolve(genes, min_fraction=min_fraction).values())
        uids = (set.intersection(*sets) if mode == "all" else set.union(*sets)) if sets else set()
        cat = self._catalog_df(validation)
        return SpatialDataCohort(cat[cat["uid"].isin(uids)], source=self._catalog._source())

    datasets_with = where_expressed  # backward-compatible alias (MCP + existing callers)

    def _catalog_df(self, validation: str | None) -> pd.DataFrame:
        cat = self._catalog.to_df()
        return cat if validation is None else cat[cat["validation_status"] == validation]

    def ranked(
        self, symbol: str, *, min_fraction: float | None = None, validation: str | None = None
    ) -> pd.DataFrame:
        """Per-dataset ``uid``, ``fraction_obs_detected``, ``total_counts`` for ``symbol``, best detection first.

        The per-dataset detection profile for ``symbol``. Filters as in :meth:`where_expressed`; one row per
        catalog dataset (max over matching features). Matches by symbol only (not Ensembl id) — for
        id-based or multi-gene membership use :meth:`where_expressed`.
        """
        df = self._df
        mask = df["symbol"].str.casefold() == symbol.casefold()
        if min_fraction is not None:
            mask &= df["fraction_obs_detected"] >= min_fraction
        cols = ["uid", "fraction_obs_detected", "total_counts"]
        hit = df.loc[mask.fillna(False).astype(bool), cols].dropna(subset=["uid"])
        hit = hit[hit["uid"].isin(set(self._catalog_df(validation)["uid"]))].groupby("uid", as_index=False).max()
        by = ["fraction_obs_detected", "total_counts"]
        return hit.sort_values(by, ascending=False, kind="stable").reset_index(drop=True)
