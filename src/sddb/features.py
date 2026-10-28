"""Cross-dataset, multi-modal feature search over the published ``feature_index.parquet``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from sddb import _catalog_schema as schema
from sddb._cache import fetch_catalog
from sddb.cohort import SpatialDataCohort

if TYPE_CHECKING:
    from pathlib import Path

    from sddb.catalog import Catalog


def sibling_url(url: str, name: str) -> str:
    """Return ``url`` with its last path component replaced by ``name``."""
    return url.rsplit("/", 1)[0] + "/" + name if "/" in url else name


def _aslist(v: object) -> list[str]:
    """List cell -> list; a null cell (schema-valid) -> empty list."""
    return [] if v is None or (not hasattr(v, "__iter__") and pd.isna(v)) else list(v)  # type: ignore[call-overload]


_BATCH_ROWS = 1_000_000


def _check_index(path: Path, url: str, kind: str) -> None:
    """Validate the index's footer schema plus a small first batch (never the whole table)."""
    name = kind.replace("_", " ")
    try:
        pf = pq.ParquetFile(path)
        head = next(pf.iter_batches(batch_size=1000), None)
        df = (head if head is not None else pf.schema_arrow.empty_table()).to_pandas()
    except Exception as err:
        raise ValueError(f"{name} at {url} is missing or corrupt: {err}") from err
    for col, dtype in (schema.GENE_INDEX_COLUMNS if kind == "gene_index" else schema.FEATURE_INDEX_COLUMNS).items():
        if col in df.columns and not dtype.startswith("list["):
            try:
                df[col] = df[col].astype(pd.api.types.pandas_dtype(dtype))
            except (TypeError, ValueError):
                pass  # left for validate() to report
    try:
        schema.validate(df, kind=kind)
    except schema.SchemaError as err:
        raise schema.SchemaError(f"{name} at {url} is invalid: {err}") from err


def _fold_match(values: pa.Array, keys: set[str]) -> dict[int, str]:
    """Positions in ``values`` (a dictionary) whose casefold is one of ``keys``, mapped to that key.

    ASCII values are matched vectorized (their casefold is their lowercase); the rare non-ASCII ones in Python.
    """
    ascii_ = pc.fill_null(pc.string_is_ascii(values), True)
    lower = pc.utf8_lower(values)
    codes = pc.indices_nonzero(pc.fill_null(pc.and_(ascii_, pc.is_in(lower, pa.array(list(keys), lower.type))), False))
    match = dict(zip(codes.to_pylist(), lower.take(codes).to_pylist(), strict=True))
    for i in pc.indices_nonzero(pc.invert(ascii_)).to_pylist():
        if (f := values[i].as_py().casefold()) in keys:
            match[i] = f
    return match


def _scan(path: Path, keys: set[str], on: tuple[str, ...], columns: list[str]) -> pd.DataFrame:
    """Rows whose ``on`` columns (string or list<string>) casefold to one of ``keys``, streamed batch by batch.

    String columns are read dictionary-encoded, so each batch casefolds only its distinct values; peak memory is
    about one batch plus the hits. Returns ``columns`` plus ``_key`` (the matched key), one row per (row, key),
    in file order.
    """
    strings = [f.name for f in pq.read_schema(path) if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    pf = pq.ParquetFile(path, read_dictionary=strings, pre_buffer=False)  # pre-buffering triples peak memory
    parts, offset = [], 0
    batches = pf.iter_batches(batch_size=_BATCH_ROWS, columns=list({*columns, *on}))
    for batch in batches if keys else ():
        pos, hit_keys = [], []
        for col in on:
            arr, parent = batch.column(col), None
            if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
                arr, parent = pc.list_flatten(arr), pc.list_parent_indices(arr)
            if not pa.types.is_dictionary(arr.type):
                arr = pc.dictionary_encode(arr)
            match = _fold_match(arr.dictionary, keys)
            if not match:
                continue
            idx = pc.indices_nonzero(pc.is_in(arr.indices, pa.array(list(match), arr.indices.type)).fill_null(False))
            pos.append((idx if parent is None else parent.take(idx)).to_numpy().astype(np.int64))
            hit_keys += [match[c] for c in arr.indices.take(idx).to_pylist()]
        if pos:
            rows = np.concatenate(pos)
            hit = batch.select(columns).take(rows)
            hit = pa.table(
                [c.dictionary_decode() if pa.types.is_dictionary(c.type) else c for c in hit.columns], names=columns
            ).to_pandas()  # plain strings: a categorical would carry the whole dictionary
            parts.append(hit.assign(_key=hit_keys, _row=rows + offset))
        offset += batch.num_rows
    if not parts:
        parts = [pf.schema_arrow.empty_table().select(columns).to_pandas().assign(_key=pd.Series(dtype=object), _row=0)]
    hits = pd.concat(parts, ignore_index=True).drop_duplicates(["_row", "_key"]).sort_values("_row", kind="stable")
    return pd.DataFrame(hits.drop(columns="_row").reset_index(drop=True))


class FeatureIndex:
    """Reader over ``feature_index.parquet`` (dataset x typed feature), bound to a Catalog.

    A query token matches a row by ``raw_label`` OR any of its ``ref_ids`` (casefolded, stripped). Queries stream
    the parquet in batches and keep only matching rows; the index is never loaded whole.
    """

    def __init__(self, catalog: Catalog, url: str | None = None) -> None:
        self._catalog = catalog
        self.url = url or sibling_url(catalog.url, "feature_index.parquet")
        self._path = fetch_catalog(self.url, cache_dir=catalog._cache_dir, refresh=catalog._refresh)
        _check_index(self._path, self.url, "feature_index")
        self._label = "raw_label"
        self._refs: tuple[str, ...] = ("ref_ids",)
        drop = ("ref_relations", "ref_confidences")
        self._dtypes = {c: t for c, t in schema.FEATURE_INDEX_COLUMNS.items() if c not in drop}

    @classmethod
    def _from_parts(
        cls, catalog: Catalog, url: str, path: Path, label: str, refs: tuple[str, ...], dtypes: dict[str, str]
    ) -> FeatureIndex:
        """A view over another index file: ``label``/``refs`` are the token columns, ``dtypes`` the returned ones."""
        self = cls.__new__(cls)
        self._catalog, self.url, self._path = catalog, url, path
        self._label, self._refs, self._dtypes = label, refs, dtypes
        return self

    # -- matching -----------------------------------------------------------------------------------

    def _hits(
        self,
        keys: set[str],
        *,
        on: tuple[str, ...] | None = None,
        feature_type: str | None = None,
        modality: str | None = None,
        min_fraction: float | None = None,
    ) -> pd.DataFrame:
        """Matching rows for casefolded ``keys`` (column ``_key``) under the type and fraction filters."""
        hit = _scan(self._path, keys, on or (self._label, *self._refs), list(self._dtypes))
        hit = schema.cast_scalars(hit, self._dtypes)
        for col, val in (("feature_type", feature_type), ("modality", modality)):
            if val is not None:
                hit = hit[(hit[col] == val).fillna(False).to_numpy(dtype=bool)]
        if min_fraction is not None:
            hit = hit[(hit["fraction_obs_detected"] >= min_fraction).fillna(False).to_numpy(dtype=bool)]
        return hit

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
        toks = self._tokens(tokens)
        hit = self._hits(
            {t.strip().casefold() for t in toks},
            feature_type=feature_type,
            modality=modality,
            min_fraction=min_fraction,
        )
        return {tok: {str(u) for u in hit.loc[hit["_key"] == tok.strip().casefold(), "uid"].dropna()} for tok in toks}

    def resolve(self, query: str, *, feature_type: str | None = None, modality: str | None = None) -> pd.DataFrame:
        """Typed hits for ``query`` (matched on ``raw_label`` or any ref id).

        One row per ``(raw_label, feature_type, modality)`` with ``ref_ids``/``ref_id_types``/``gene_xref`` of the
        first matching row, the sorted dataset ``uids`` and ``n_datasets``. Empty frame when nothing matches.
        """
        hit = self._hits({query.strip().casefold()}, feature_type=feature_type, modality=modality)
        cols = ["raw_label", "feature_type", "modality", "ref_ids", "ref_id_types", "gene_xref", "uids", "n_datasets"]
        if hit.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        for (label, ftype, mod), g in hit.groupby(["raw_label", "feature_type", "modality"], dropna=False, sort=True):
            first = g.iloc[0]
            uids = sorted({str(u) for u in g["uid"].dropna()})
            rows.append(
                (
                    label,
                    ftype,
                    mod,
                    _aslist(first["ref_ids"]),
                    _aslist(first["ref_id_types"]),
                    _aslist(first["gene_xref"]),
                    uids,
                    len(uids),
                )
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
        lab = self._hits({key}, on=(self._label,))
        lab = lab[lab["feature_type"].isin(["gene", "gene_activity"]).to_numpy(dtype=bool)]
        ids = {key} | {str(v).casefold() for cell in lab["gene_xref"] for v in _aslist(cell)}
        hit = pd.concat([lab, self._hits(ids, on=("gene_xref",))])
        if min_fraction is not None:
            hit = hit[(hit["fraction_obs_detected"] >= min_fraction).fillna(False).to_numpy(dtype=bool)]
        return self._cohort({str(u) for u in hit["uid"].dropna()}, validation)

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
        hit = self._hits(
            {entity.strip().casefold()}, feature_type=feature_type, modality=modality, min_fraction=min_fraction
        )
        cols = ["uid", "fraction_obs_detected", "total_counts"]
        hit = hit[cols].dropna(subset=["uid"])
        hit = hit[hit["uid"].isin(set(self._catalog_df(validation)["uid"]))].groupby("uid", as_index=False).max()
        by = ["fraction_obs_detected", "total_counts"]
        return hit.sort_values(by, ascending=False, kind="stable").reset_index(drop=True)
