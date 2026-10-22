# Catalog columns

Every column of the catalog schema. "Facet" columns are categorical filters (`Catalog.facets()`); "range" columns are numeric and filter with `__gte` / `__lte`.

```{include} _catalog_columns.md
```

## Study and condition columns

All optional: older catalogs without them still load, and they are null when the source has no such metadata.

### Dataset columns

| Column | Description |
|---|---|
| `study_id` | Identifier of the study the dataset belongs to. |
| `sample_title` | Free-text title of the individual sample. |
| `timepoint` | Sampling timepoint as written in the source (e.g. `day 7`). |
| `timepoint_kind` | Kind of timepoint (e.g. days post injury, age, treatment duration). |
| `timepoint_days` | Timepoint normalised to days (numeric; supports range filters). |
| `treatment` | Treatment or perturbation applied to the sample. |
| `genotype` | Genotype of the sample. |
| `condition` | Condition label for the sample (e.g. control vs. disease). |

### Collection columns (`collections.parquet`)

| Column | Description |
|---|---|
| `overall_design` | Study-level design summary text. |
| `pubmed_id` | PubMed ID of the associated paper. |
| `source_url` | URL of the study's source record. |
| `parent` | Slug of the parent collection, if any. |
| `source` | Origin of the collection (e.g. GEO, HEST). |
| `year` | Publication year (integer). |
| `contrast_axes` | JSON object `{axis: {value: count}}` of the condition axes that vary across members. |
| `assays` | Distinct assays across member datasets. |
| `modalities` | Distinct modalities across member datasets. |
| `organisms` | Distinct organisms across member datasets. |
| `tissues` | Distinct tissues across member datasets. |
| `diseases` | Distinct diseases across member datasets. |
| `timepoint_kinds` | Distinct timepoint kinds across member datasets. |
| `condition_values` | Distinct condition values across member datasets. |
