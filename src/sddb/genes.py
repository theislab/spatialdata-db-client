"""Cross-dataset gene search over the published gene index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from sddb import _catalog_schema as schema
from sddb._cache import fetch_catalog
from sddb.cohort import SpatialDataCohort
from sddb.features import FeatureIndex, _check_index, sibling_url

if TYPE_CHECKING:
    from sddb.catalog import Catalog


class GeneIndex:
    """Reader over ``gene_index.parquet``: a gene-only view over :class:`~sddb.features.FeatureIndex`."""

    def __init__(self, catalog: Catalog, url: str | None = None) -> None:
        self._catalog = catalog
        self.url = url or sibling_url(catalog.url, "gene_index.parquet")
        path = fetch_catalog(self.url, cache_dir=catalog._cache_dir, refresh=catalog._refresh)
        strings = _check_index(path, self.url, "gene_index")
        self._fi = FeatureIndex._from_parts(
            catalog, self.url, path, strings, "symbol", ("feature_id",), schema.GENE_INDEX_COLUMNS
        )

    def resolve(self, tokens: str | list[str], *, min_fraction: float | None = None) -> dict[str, set[str]]:
        """Map each token to the uids of datasets expressing it.

        A token (casefolded, stripped) matches a gene-index row by symbol OR Ensembl ``feature_id``;
        an ambiguous symbol therefore unions all its feature ids. Unmatched tokens map to an empty set.
        Keys keep the original token spelling.
        """
        return self._fi._uid_sets(tokens, min_fraction=min_fraction)

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
        return self._fi.where_measured(genes, mode=mode, min_fraction=min_fraction, validation=validation)

    datasets_with = where_expressed  # backward-compatible alias (MCP + existing callers)

    def _catalog_df(self, validation: str | None) -> pd.DataFrame:
        return self._fi._catalog_df(validation)

    def ranked(self, symbol: str, *, min_fraction: float | None = None, validation: str | None = None) -> pd.DataFrame:
        """Per-dataset ``uid``, ``fraction_obs_detected``, ``total_counts`` for ``symbol``, best detection first.

        The per-dataset detection profile for ``symbol``. Filters as in :meth:`where_expressed`; one row per
        catalog dataset (max over matching features). ``symbol`` may be a symbol or an Ensembl feature id.
        """
        return self._fi.ranked(symbol, min_fraction=min_fraction, validation=validation)
