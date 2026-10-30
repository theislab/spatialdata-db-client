# Cite and license

## Cite the datasets

Every dataset belongs to a study. A cohort writes one BibTeX entry for each study it contains:

```python
from sddb import Catalog
from sddb.citations import citation_entries

cohort = Catalog().query(technology="Visium", tissue="lung")
cohort.citations("refs.bib")          # write BibTeX to a file
bibtex, n_entries = cohort.citations_text()  # or get it as a string
bibtex, n_published, n_fallback = citation_entries(cohort)  # with the two kinds counted apart
```

A study whose `study_id` is a key in the catalog's published `citations.bib` gets that entry. Not every study has one yet. Each study without one, and each dataset without a `study_id`, gets a fallback `@misc` entry instead, so the file cites every study in the cohort. The datasets of one study share one fallback entry. It looks like this:

```bibtex
@misc{10x:visium-cytassist-human-colon-cancer-p2,
  title = {visium-cytassist-human-colon-cancer-p2},
  howpublished = {\url{https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-colon-qmnnmcp2b2w5ddyn0000/}},
  doi = {10.1038/s41588-025-02193-3},
  year = {2025},
  note = {SpatialData-DB dataset qMnnMcP2B2W5ddYN0000; license CC-BY-4.0}
}
```

The key is the `study_id`, or the uid when the dataset has no `study_id`. The title is the dataset title and the link points to the dataset page on the website (the first dataset of the study, by uid). `doi` appears when the catalog has one. `year` comes from `publication_year`, or else from `publication_date` or `published_at`. The note lists the uids (the first five, then a count) and their licenses. When the cohort needs any fallback entries, one warning reports the counts, for example "3 of 12 studies have a published citation; added 9 fallback entries that cite the dataset page; check them before publishing". Check those entries, and replace them with the paper's own entry where there is one, before you publish. An empty cohort raises `ValueError` instead of writing an empty file.

From the shell, `sddb-cli cite` takes the same facets as a query:

```bash
sddb-cli cite --technology Visium --tissue lung -o refs.bib
```

It prints how many entries it wrote and how many of them are fallback entries, for example "wrote 12 citations (9 fallback) -> refs.bib".

See {doc}`reference/cli` for all options.

## Cite the software

The repository ships a `CITATION.cff`. GitHub's "Cite this repository" button reads it, and so do Zotero and other reference managers.

## Cite the database

The paper describing the database is forthcoming. This page will link it once it is out.

## Licenses

Each dataset keeps the license of its source. The catalog records it in these columns (see {doc}`reference/catalog-columns`):

- `license`: license identifier, for example `CC-BY-4.0`
- `license_url`: link to the license text
- `license_noncommercial`: true if the license forbids commercial use
- `license_redistributable`: true if the license allows redistribution; false when it does not, and false for every dataset whose license is unknown
- `license_unknown`: true if no license could be determined

`Catalog.query` can filter on them:

```python
cat = Catalog()
cat.query(license_set=True)                       # keep only datasets whose license is known
cat.query(noncommercial=False)                    # only datasets cleared for commercial use
cat.query(license="CC-BY-4.0")                    # one exact license id
```

`noncommercial=False` keeps a dataset only when its license is known and does not forbid commercial use. It returns the same set as "Commercial use OK" on the website. Datasets with an unknown license are left out, because their terms are not known. This changed in this release; before, `noncommercial=False` also returned unknown-license datasets. `noncommercial=True` keeps only NonCommercial data, and the default `None` applies no license filter. `license_set=True` on its own keeps every known license, including NonCommercial ones.

### License glossary

These are the `license` values in the catalog of 2026-10-28 (6087 datasets):

| `license` | Datasets | Commercial use | Meaning |
| --- | ---: | --- | --- |
| `ncbi-geo-policy` | 3530 | allowed | Data deposited in GEO. GEO data: NCBI places no restrictions on use; submitters may hold copyright or other rights. |
| `CC-BY-NC-SA-4.0` | 1072 | no | Creative Commons Attribution-NonCommercial-ShareAlike 4.0. Credit the authors, do not use commercially, and share adaptations under the same license. |
| `unknown` | 1031 | not cleared | No license could be determined. `license_unknown` is true. |
| `CC-BY-4.0` | 333 | allowed | Creative Commons Attribution 4.0. Credit the authors. |
| `allen-institute-tou` | 115 | no | Allen Institute terms of use. These are noncommercial, so `license_noncommercial` is true. |
| `CC0-1.0` | 6 | allowed | Creative Commons public domain dedication. No conditions. |

The three boolean columns mean:

- `license_noncommercial` is true when the license forbids commercial use (`CC-BY-NC-SA-4.0` and `allen-institute-tou`).
- `license_redistributable` is true when the license allows redistribution. It is false only for `unknown`, so false means the terms are not known, not that redistribution is forbidden.
- `license_unknown` is true when no license could be determined. Those rows have `license` set to `unknown` and `license_noncommercial` false, so check `license_unknown` before you treat the latter as permission.

The catalog lists every dataset it serves, including the 1031 whose license is unknown (17% of 6087). The default `license_set=None` applies no license filter, so a search does not silently hide them. Pass `license_set=True` to keep only datasets with a known license, or `noncommercial=False` to keep only those cleared for commercial use. Check the license of your final selection before you redistribute or publish derived data.
