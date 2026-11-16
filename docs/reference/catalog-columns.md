# Catalog columns

Every column of the catalog schema. "Facet" columns are categorical filters (`Catalog.query(<column>=...)`, values from `sddb-cli facets <column>`); "range" columns are numeric and filter with `__gte` / `__lte`. `source` is not stored in the catalog file: the client derives it on load.

## Facet values

- A string facet value is first looked up as given in that column of the whole catalog. If it is not there, case, spaces, hyphens and underscores are ignored. Since schema 3, `technology` holds a vendor-prefixed id such as `10x_visium`, `10x_visium_hd`, `10x_xenium` or `vizgen_merscope` (labels in the release's `technology.json`, see `Catalog().technologies()`). A `technology` value may also be an `assay` label (`technology="10x Xenium"` selects `10x_xenium`) or the id without its vendor prefix, which is the schema-2 name (`technology="Visium"` selects `10x_visium`, `"VisiumHD"` selects `10x_visium_hd`, `"Merscope"` selects `vizgen_merscope`). A value that still does not resolve raises `ValueError` with the closest known value ("did you mean '10x_xenium'?"). A known value that matches nothing after the other filters gives an empty cohort.
- `hne_image` and `if_image` take `True` or `False`. A null value means unknown and matches neither.
- `source` is the lowercased `study_id` prefix before ":". In the 2026-10-28 catalog (6087 datasets) the values are `geo` (4113), `hest` (1070), `10x` (175), `abc` (115), `bia` (4), `metaspace` (3), `bruker` (3), `doi` (2) and `figshare` (2). 600 datasets have no `study_id`, so their source is null.
- `query(redistributable=True)` keeps datasets whose license is known and allows redistribution (`license_redistributable` is true and `license_unknown` is false). `redistributable=False` keeps the rest, including every dataset with an unknown license, where false means the terms are not known. Every dataset in the catalog is served, whatever its license.

```{include} _catalog_columns.md
```

## Quality columns

| Column | Description |
|---|---|
| `validation_status` | Result of schema validation, `pass` or `fail`. In the current catalog every dataset passes validation, so this filter changes nothing yet. Filter on `tier` (`silver`, `bronze`) for quality. |
| `tier` | Curation tier: `gold`, `silver` or `bronze`, from most to least curated. Some datasets have no tier yet. |
| `non_integer_counts` | True when a count table is stored as deposited with fractional values (mostly x.5 or x.25). The values aren't rounded, and the depositor doesn't state the cause. |

## Dates

- `publication_date` is the source release date: the day the repository (GEO, Zenodo, BioImage Archive, ...) made the data public. It isn't the paper's date, and for a GEO series it is often a year earlier.
- `publication_year` is the year of the paper that describes the dataset (null when no paper is linked). Use it to cite, and `doi`/`pmid` for the paper itself.
- The names stay until the next catalog schema version.

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
