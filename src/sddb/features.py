"""Cross-dataset, multi-modal feature search over the published ``feature_index.parquet``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sddb import _catalog_schema as schema
from sddb._cache import fetch_catalog
from sddb.cohort import SpatialDataCohort

if TYPE_CHECKING:
    from sddb.catalog import Catalog

def sibling_url(url: str, name: str) -> str:
    """Return ``url`` with its last path component replaced by ``name``."""
    return url.rsplit("/", 1)[0] + "/" + name if "/" in url else name


def _aslist(v: object) -> list[str]:
    """List cell -> list; a null cell (schema-valid) -> empty list."""
    return [] if v is None or (not hasattr(v, "__iter__") and pd.isna(v)) else list(v)  # type: ignore[call-overload]


def _explode_fold(series: pd.Series) -> pd.Series:
    """Casefolded one-value-per-row view of a list column (index = originating row label)."""
    return series.explode().dropna().astype(str).str.casefold()


class FeatureIndex:
    """Reader over ``feature_index.parquet`` (dataset x typed feature), bound to a Catalog.

    A query token matches a row by ``raw_label`` OR any of its ``ref_ids`` (casefolded, stripped).
    """

    def __init__(self, catalog: Catalog, url: str | None = None) -> None:
        self._catalog = catalog
        self.url = url or sibling_url(catalog.url, "feature_index.parquet")
        path = fetch_catalog(self.url, cache_dir=catalog._cache_dir, refresh=catalog._refresh)
        try:
            df = pd.read_parquet(path)
        except Exception as err:
            raise ValueError(f"feature index at {self.url} is missing or corrupt: {err}") from err
        for col, dtype in schema.FEATURE_INDEX_COLUMNS.items():
            if col in df.columns and not dtype.startswith("list["):
                try:
                    df[col] = df[col].astype(pd.api.types.pandas_dtype(dtype))
                except (TypeError, ValueError):
                    pass  # left for validate() to report
        try:
            schema.validate(df, kind="feature_index")
        except schema.SchemaError as err:
            raise schema.SchemaError(f"feature index at {self.url} is invalid: {err}") from err
        df = df.reset_index(drop=True)
        self._init_parts(df, df["raw_label"].str.casefold(), _explode_fold(df["ref_ids"]), _explode_fold(df["gene_xref"]))

    def _init_parts(self, df: pd.DataFrame, lab: pd.Series, ref: pd.Series, xref: pd.Series) -> None:
        self._df = df
        self._lab = lab  # casefolded raw_label per row (may hold pd.NA)
        self._ref = ref  # casefolded ref ids, one per (row, id)
        self._xref = xref  # casefolded Ensembl gene ids, one per (row, id)

    @classmethod
    def _from_parts(
        cls, catalog: Catalog, url: str, df: pd.DataFrame, lab: pd.Series, ref: pd.Series, xref: pd.Series
    ) -> FeatureIndex:
        self = cls.__new__(cls)
        self._catalog = catalog
        self.url = url
        self._init_parts(df, lab, ref, xref)
        return self

    # -- matching -----------------------------------------------------------------------------------

    def _rows(self, key: str, *, feature_type: str | None, modality: str | None) -> np.ndarray:
        """Boolean row mask for rows labelled/referenced as ``key`` (a casefolded token) under the type filters."""
        df = self._df
        mask = (self._lab == key).fillna(False).to_numpy(dtype=bool)
        mask |= df.index.isin(self._ref.index[(self._ref == key).to_numpy()])
        return mask & self._type_mask(feature_type, modality)

    def _type_mask(self, feature_type: str | None, modality: str | None) -> np.ndarray:
        df = self._df
        mask = np.ones(len(df), dtype=bool)
        if feature_type is not None:
            mask &= (df["feature_type"] == feature_type).fillna(False).to_numpy(dtype=bool)
        if modality is not None:
            mask &= (df["modality"] == modality).fillna(False).to_numpy(dtype=bool)
        return mask

    def _frac_mask(self, min_fraction: float | None) -> np.ndarray:
        if min_fraction is None:
            return np.ones(len(self._df), dtype=bool)
        return (self._df["fraction_obs_detected"] >= min_fraction).fillna(False).to_numpy(dtype=bool)

    @staticmethod
    def _tokens(tokens: str | list[str]) -> list[str]:
        return [tokens] if isinstance(tokens, str) else list(tokens)

    def _uid_sets(
        self,
        tokens: str | list[str],
        *,
        feature_type: str | None = None,
        modality: str | None = None,
        min_fraction: float | None = None,
    ) -> dict[str, set[str]]:
        frac = self._frac_mask(min_fraction)
        out: dict[str, set[str]] = {}
        for tok in self._tokens(tokens):
            mask = self._rows(tok.strip().casefold(), feature_type=feature_type, modality=modality) & frac
            out[tok] = {str(u) for u in self._df.loc[mask, "uid"].dropna()}
        return out

    def resolve(self, query: str, *, feature_type: str | None = None, modality: str | None = None) -> pd.DataFrame:
        """Typed hits for ``query`` (matched on ``raw_label`` or any ref id).

        One row per ``(raw_label, feature_type, modality)`` with ``ref_ids``/``ref_id_types``/``gene_xref`` of the
        first matching row, the sorted dataset ``uids`` and ``n_datasets``. Empty frame when nothing matches.
        """
        mask = self._rows(query.strip().casefold(), feature_type=feature_type, modality=modality)
        hit = self._df.loc[mask]
        cols = ["raw_label", "feature_type", "modality", "ref_ids", "ref_id_types", "gene_xref", "uids", "n_datasets"]
        if hit.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        for (label, ftype, mod), g in hit.groupby(["raw_label", "feature_type", "modality"], dropna=False, sort=True):
            first = g.iloc[0]
            uids = sorted({str(u) for u in g["uid"].dropna()})
            rows.append(
                (label, ftype, mod, _aslist(first["ref_ids"]), _aslist(first["ref_id_types"]), _aslist(first["gene_xref"]),
                 uids, len(uids))
            )
        return pd.DataFrame(rows, columns=cols)

    # -- dataset-level queries ----------------------------------------------------------------------

    def _catalog_df(self, validation: str | None) -> pd.DataFrame:
        cat = self._catalog.to_df()
        return cat if validation is None else cat[cat["validation_status"] == validation]

    def _cohort(self, uids: set[str], validation: str | None) -> SpatialDataCohort:
        cat = self._catalog_df(validation)
        return SpatialDataCohort(cat[cat["uid"].isin(uids)], source=self._catalog._source())

    def where_measured(
        self,
        entity: str | list[str],
        *,
        mode: str = "all",
        feature_type: str | None = None,
        modality: str | None = None,
        min_fraction: float | None = None,
        validation: str | None = None,
    ) -> SpatialDataCohort:
        """Catalog datasets measuring ``entity`` (label or reference id, case-insensitive), as a cohort.

        ``mode`` is ``"all"`` (every entity, AND; default) or ``"any"`` (union). ``feature_type``/``modality``
        restrict the matched rows; ``min_fraction`` keeps rows with ``fraction_obs_detected >= min_fraction``;
        ``validation`` keeps one ``validation_status`` (default ``None`` includes bronze/non-pass).
        """
        if mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', got {mode!r}")
        sets = list(
            self._uid_sets(entity, feature_type=feature_type, modality=modality, min_fraction=min_fraction).values()
        )
        uids = (set.intersection(*sets) if mode == "all" else set.union(*sets)) if sets else set()
        return self._cohort(uids, validation)

    def by_gene(
        self, gene: str, *, min_fraction: float | None = None, validation: str | None = None
    ) -> SpatialDataCohort:
        """Datasets relating to ``gene`` in ANY modality (RNA, protein targeting it, gene activity).

        ``gene`` is a symbol or Ensembl id. A symbol is first resolved to Ensembl ids through gene/gene_activity
        rows; datasets are then matched on ``gene_xref`` (plus the gene/gene_activity rows labelled with it).
        """
        key = gene.strip().casefold()
        df = self._df
        gene_like = df["feature_type"].isin(["gene", "gene_activity"]).fillna(False).to_numpy(dtype=bool)
        label_mask = (self._lab == key).fillna(False).to_numpy(dtype=bool) & gene_like
        ids = {key} | set(self._xref[df.index[label_mask].intersection(self._xref.index)])
        mask = label_mask | df.index.isin(self._xref.index[self._xref.isin(ids).to_numpy()])
        mask &= self._frac_mask(min_fraction)
        return self._cohort({str(u) for u in df.loc[mask, "uid"].dropna()}, validation)

    def ranked(
        self,
        entity: str,
        *,
        feature_type: str | None = None,
        modality: str | None = None,
        min_fraction: float | None = None,
        validation: str | None = None,
    ) -> pd.DataFrame:
        """Per-dataset ``uid``, ``fraction_obs_detected``, ``total_counts`` for ``entity``, best detection first."""
        mask = self._rows(entity.strip().casefold(), feature_type=feature_type, modality=modality)
        mask &= self._frac_mask(min_fraction)
        cols = ["uid", "fraction_obs_detected", "total_counts"]
        hit = self._df.loc[mask, cols].dropna(subset=["uid"])
        hit = hit[hit["uid"].isin(set(self._catalog_df(validation)["uid"]))].groupby("uid", as_index=False).max()
        by = ["fraction_obs_detected", "total_counts"]
        return hit.sort_values(by, ascending=False, kind="stable").reset_index(drop=True)
