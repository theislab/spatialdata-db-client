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
    # license (vocab id + properties derived by the producer from the License ULabel)
    "license": "string",
    "license_url": "string",
    "license_noncommercial": "boolean",
    "license_redistributable": "boolean",
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
    # presentation / curation superset — consumed by the website build; the pip
    # client carries them through to_df() but does not require them. chemistry_version
    # (reagent-kit version) is a SEPARATE attribute from technology_version above.
    "slug": "string",
    "title": "string",
    "pathological": "boolean",
    "product": "string",
    "modality": "string",
    "biomaterial_type": "string",
    "chemistry_version": "string",
    "staining_method": "list[string]",
    "preservation_method": "string",
    "hne_image": "boolean",
    "if_image": "boolean",
    "ftu_annotation": "boolean",
    "publication_date": "string",
    "sample_id": "string",
    "donor_id": "string",
    "database_version": "string",
    "sddb_id": "string",
    "disease_details": "string",
    "default_table": "string",
    "dataset_url": "string",
    "thumbnail_url": "string",
    "collections": "list[string]",
}

REQUIRED_COLUMNS: tuple[str, ...] = ("uid", "technology", "assay", "organism", "validation_status", "zarr_url")

# Collection-level table (one row per ln.Collection), published as collections.parquet
# beside catalog.parquet. The website renders its collection pages from this.
COLLECTION_COLUMNS: dict[str, str] = {
    "slug": "string",
    "key": "string",
    "name": "string",
    "kind": "string",
    "description": "string",
    "reference": "string",
    "reference_type": "string",
    "assay": "string",
    "organism": "string",
    "tissue": "string",
    "n_datasets": "Int64",
    "n_members_total": "Int64",
    "size_bytes": "Int64",
    "members": "list[string]",
}
COLLECTION_REQUIRED: tuple[str, ...] = ("slug", "name", "members")

FACETS: tuple[str, ...] = (
    "organism",
    "tissue",
    "disease",
    "development_stage",
    "technology",
    "assay",
    "tier",
    "validation_status",
    "license",
    "license_noncommercial",
)

# numeric columns that accept range operators (col__gte / __gt / __lte / __lt) in Catalog.query:
# exactly the CATALOG_COLUMNS entries whose dtype is Int64/Float64 (derived, so the two stay in sync).
RANGE_COLUMNS: tuple[str, ...] = tuple(c for c, t in CATALOG_COLUMNS.items() if t in ("Int64", "Float64"))

GENE_INDEX_COLUMNS: dict[str, str] = {
    "symbol": "string",
    "feature_id": "string",
    "uid": "string",
    "total_counts": "Float64",
    "fraction_obs_detected": "Float64",
}


class SchemaError(ValueError):
    """Raised when a catalog/gene-index DataFrame violates the schema contract."""


def _is_null(v) -> bool:
    """A single cell is null — scalar NaN/NA/None. A list/array cell is never null."""
    return v is None or (not hasattr(v, "__iter__") and pd.isna(v))


def _all_null(series: pd.Series) -> bool:
    """True if every cell is null — list/array-safe (avoids Series.isna() ambiguity on list cells)."""
    return all(_is_null(v) for v in series)


def cast_scalars(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Cast ``df``'s scalar columns to their declared dtypes, leaving list<string> columns as object."""
    scalar = {c: t for c, t in columns.items() if c in df.columns and not t.startswith("list")}
    return df.astype(scalar)


def _compatible(series: pd.Series, dtype: str) -> bool:
    """Return whether ``series`` can be read as the declared pandas dtype."""
    if dtype == "list[string]":
        # parquet list<string> -> object column whose cells are None or a list/array of str.
        return pdt.is_object_dtype(series.dtype) and all(
            _is_null(v) or (not isinstance(v, str) and hasattr(v, "__iter__") and all(isinstance(e, str) for e in v))
            for v in series
        )
    if pdt.is_bool_dtype(series.dtype) and not pdt.is_object_dtype(series.dtype):
        return dtype == "boolean"
    if dtype == "string":
        if pdt.is_string_dtype(series.dtype) and not pdt.is_object_dtype(series.dtype):
            return True  # pandas StringDtype or a pyarrow-backed string ArrowDtype
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
        checks ``GENE_INDEX_COLUMNS`` (all required); ``"collections"`` checks
        ``COLLECTION_COLUMNS``/``COLLECTION_REQUIRED``.

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
    elif kind == "collections":
        columns, required = COLLECTION_COLUMNS, COLLECTION_REQUIRED
    else:
        raise ValueError(f"unknown kind {kind!r}; expected 'catalog', 'gene_index' or 'collections'")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")
    for col, dtype in columns.items():
        if col in df.columns and not _compatible(df[col], dtype):
            raise SchemaError(f"column {col!r} has dtype {df[col].dtype}, incompatible with {dtype}")
    for col in required:
        if len(df) and _all_null(df[col]):
            raise SchemaError(f"required column {col!r} is all-null")
