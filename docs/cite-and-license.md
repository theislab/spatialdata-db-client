# Cite and license

## Cite the datasets

Every dataset belongs to a study. A cohort writes the BibTeX entries of the studies it contains:

```python
from sddb import Catalog

cohort = Catalog().query(technology="Visium", tissue="lung")
cohort.citations("refs.bib")          # write BibTeX to a file
bibtex, n_entries = cohort.citations_text()  # or get it as a string
```

From the shell, `sddb-cli cite` takes the same facets as a query:

```bash
sddb-cli cite --technology Visium --tissue lung -o refs.bib
```

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
- `license_redistributable`: true if the data may be redistributed
- `license_unknown`: true if no license could be determined

`Catalog.query` can filter on them:

```python
cat = Catalog()
cat.query(license_set=True)                       # keep only datasets whose license is known
cat.query(noncommercial=False)                    # only datasets cleared for commercial use
cat.query(license_set=True, noncommercial=False)  # both
cat.query(license="CC-BY-4.0")                    # one exact license id
```

`noncommercial=True` keeps only NonCommercial data. Check the license of your final selection before you redistribute or publish derived data.
