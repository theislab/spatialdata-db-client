"""Catalog schema contract shared by the client and the engine's producer."""

from __future__ import annotations

from typing import Any

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
    # All modalities present in the dataset (multimodal); scalar `modality` kept for back-compat.
    "modalities": "list[string]",
    "biomaterial_type": "string",
    "chemistry_version": "string",
    "staining_method": "list[string]",
    "preservation_method": "string",
    "hne_image": "boolean",
    "if_image": "boolean",
    "ftu_annotation": "boolean",
    "mpp": "Float64",
    "image_width": "Int64",
    "image_height": "Int64",
    "magnification": "string",
    # Plain-text `print(sdata)` of the converted SpatialData object (store path stripped).
    "sdata_repr": "string",
    # Raw row count of the transcripts points element (imaging techs; null without points).
    "n_transcripts": "Int64",
    # Visium HD capture-area size ("6.5 mm" / "11 mm"); fixes the bin grid (838² vs 1381² at 8 µm).
    "capture_area": "string",
    "publication_date": "string",
    # Publication verification (set by the engine's publication-verification gate): the resolved
    # paper identifiers + the verify status (verified / provisional / no_associated_publication).
    "doi": "string",
    "pmid": "string",
    "publication_year": "string",
    "publication_verify_status": "string",
    "sample_id": "string",
    "donor_id": "string",
    "database_version": "string",
    "sddb_id": "string",
    "disease_details": "string",
    "default_table": "string",
    "dataset_url": "string",
    "thumbnail_url": "string",
    "collections": "list[string]",
    # Per-sample study / condition metadata (optional; null when the source has none).
    "sample_title": "string",
    "timepoint": "string",
    "timepoint_kind": "string",
    "timepoint_days": "Float64",
    "treatment": "string",
    "genotype": "string",
    "condition": "string",
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
    "abstract": "string",
    "abstract_license": "string",
    "abstract_source": "string",
    "study_title": "string",
    "study_authors": "string",
    "study_year": "string",
    "study_journal": "string",
    # Study-level collection metadata + facet roll-ups over members (all optional).
    "overall_design": "string",
    "pubmed_id": "string",
    "source_url": "string",
    "parent": "string",
    "source": "string",
    "year": "Int64",
    "contrast_axes": "string",  # JSON object {axis: {value: count}}
    "assays": "list[string]",
    "modalities": "list[string]",
    "organisms": "list[string]",
    "tissues": "list[string]",
    "diseases": "list[string]",
    "timepoint_kinds": "list[string]",
    "condition_values": "list[string]",
}
COLLECTION_REQUIRED: tuple[str, ...] = ("slug", "name", "members")

FACETS: tuple[str, ...] = (
    "study_id",
    "doi",
    "organism",
    "tissue",
    "disease",
    "development_stage",
    "technology",
    "assay",
    "modalities",
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

# Typed feature index (one row per dataset x feature). Parallel ref_* arrays (not nested structs)
# keep it parquet/client-friendly; gene_xref carries the gene ids a non-gene feature maps to.
FEATURE_INDEX_COLUMNS: dict[str, str] = {
    "raw_label": "string",
    "feature_type": "string",
    "modality": "string",
    "uid": "string",
    "total_counts": "Float64",
    "fraction_obs_detected": "Float64",
    "ref_ids": "list[string]",
    "ref_id_types": "list[string]",
    "ref_relations": "list[string]",
    "ref_confidences": "list[double]",
    "gene_xref": "list[string]",
}

# Typed-search table (one row per searchable entity); bitset is a packed dataset-membership bitmap.
FEATURE_SEARCH_COLUMNS: dict[str, str] = {
    "entity_id": "string",
    "entity_type": "string",
    "id_type": "string",
    "display": "string",
    "feature_types": "list[string]",
    "n_datasets": "int32",
    "bitset": "binary",
}


class SchemaError(ValueError):
    """Raised when a catalog/gene-index DataFrame violates the schema contract."""


def _is_null(v: Any) -> bool:
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
