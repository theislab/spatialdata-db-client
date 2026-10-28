# spatialdata-db

{{ hero_line }}

```bash
pip install git+https://github.com/theislab/spatialdata-db-client
```

Needs Python 3.12 or newer. The package isn't on PyPI yet, so install it from GitHub.

```python
from sddb import Catalog

hits = Catalog().query(technology="Visium", tissue="lung")
hits.summary()
smallest = min(hits, key=lambda ds: ds.size_bytes)
sdata = smallest.to_sdata()  # ~10 MB, copied to the local cache once
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Coming from the website?
:link: start
:link-type: doc
Take the uid or slug from a page on the DB site and load it in Python.
:::

:::{grid-item-card} Find & load
:link: guides/find-and-load
:link-type: doc
Filter the catalog by facets or free text and load datasets.
:::

:::{grid-item-card} Cohorts
:link: guides/cohorts
:link-type: doc
Freeze, split and reproduce sets of datasets.
:::

:::{grid-item-card} Agents & CLI
:link: guides/cli-and-agents
:link-type: doc
Use the catalog from the shell and from AI agents.
:::

:::{grid-item-card} API
:link: reference/api
:link-type: doc
Full reference for the Python API.
:::
::::

Using data from the DB? See {doc}`cite-and-license`.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

start
intro/assays
intro/db-stack
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Guides

guides/find-and-load
guides/cohorts
guides/ml
guides/visium-atlas/visium-atlas
guides/cli-and-agents
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/api
reference/cli
reference/mcp
reference/catalog-columns
```

```{toctree}
:hidden:
:maxdepth: 1

cite-and-license
changelog
```
