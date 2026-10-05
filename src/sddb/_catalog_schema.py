"""Catalog schema contract shared by the client and the engine's producer."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

CATALOG_COLUMNS: dict[str, str] = {
    # identity
    "uid": "string",
    "canonical_key": "string",
    "study_id": "string",
    "collection_name": "string",
    "issue_url": "string",
    # biology
    "organism": "string",
    "organism_id": "string",
    "tissue": "string",
    "tissue_id": "string",
    "disease": "string",
    "disease_id": "string",
    "development_stage": "string",
    # technology
    "technology": "string",
    "assay": "string",
    "technology_version": "string",
    "panel_size": "Int64",
    # validation
    "validation_status": "string",
    "tier": "string",
    # license
    "license_spdx": "string",
    "license_unknown": "boolean",
    # headline stats
    "n_obs": "Int64",
    "n_features": "Int64",
    "total_counts": "Float64",
    # access
    "zarr_url": "string",
    "card_url": "string",
    "vitessce_url": "string",
    "size_bytes": "Int64",
    # provenance
    "container_tag": "string",
    "sddb_version": "string",
    "published_at": "string",
}

REQUIRED_COLUMNS: tuple[str, ...] = ("uid", "technology", "assay", "organism", "validation_status", "zarr_url")

FACETS: tuple[str, ...] = (
    "organism",
    "tissue",
    "disease",
    "development_stage",
    "technology",
    "assay",
    "tier",
    "validation_status",
    "license_spdx",
)

GENE_INDEX_COLUMNS: dict[str, str] = {
    "symbol": "string",
    "feature_id": "string",
    "uid": "string",
    "total_counts": "Float64",
    "fraction_obs_detected": "Float64",
}


class SchemaError(ValueError):
    """Raised when a catalog/gene-index DataFrame violates the schema contract."""


def _compatible(series: pd.Series, dtype: str) -> bool:
    """Return whether ``series`` can be read as the declared pandas dtype."""
    if pdt.is_bool_dtype(series.dtype) and not pdt.is_object_dtype(series.dtype):
        return dtype == "boolean"
    if dtype == "string":
        if isinstance(series.dtype, pd.StringDtype):
            return True
        return pdt.is_object_dtype(series.dtype) and all(isinstance(v, str) for v in series.dropna())
    if dtype == "Int64":
        return pdt.is_integer_dtype(series.dtype)
    if dtype == "Float64":
        return pdt.is_float_dtype(series.dtype) or pdt.is_integer_dtype(series.dtype)
    if dtype == "boolean":
        return pdt.is_object_dtype(series.dtype) and all(isinstance(v, bool) for v in series.dropna())
    return False


def validate(df: pd.DataFrame, *, kind: str = "catalog") -> None:
    """Validate a catalog or gene-index DataFrame against the contract.

    Unknown extra columns are allowed.

    Parameters
    ----------
    df
        Frame to validate.
    kind
        ``"catalog"`` checks ``CATALOG_COLUMNS``/``REQUIRED_COLUMNS``; ``"gene_index"``
        checks ``GENE_INDEX_COLUMNS`` (all required).

    Raises
    ------
    SchemaError
        On a missing required column, a known column with an incompatible dtype,
        or an all-null required column.
    """
    if kind == "catalog":
        columns, required = CATALOG_COLUMNS, REQUIRED_COLUMNS
    elif kind == "gene_index":
        columns, required = GENE_INDEX_COLUMNS, tuple(GENE_INDEX_COLUMNS)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected 'catalog' or 'gene_index'")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")
    for col, dtype in columns.items():
        if col in df.columns and not _compatible(df[col], dtype):
            raise SchemaError(f"column {col!r} has dtype {df[col].dtype}, incompatible with {dtype}")
    for col in required:
        if len(df) and df[col].isna().all():
            raise SchemaError(f"required column {col!r} is all-null")
