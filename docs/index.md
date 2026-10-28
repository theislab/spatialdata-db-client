# spatialdata-db: spatial omics datasets, ready to load as SpatialData

spatialdata-db is a catalog of public spatial omics datasets, from Visium and Xenium to MERSCOPE, CosMx and mass spectrometry imaging.
Each dataset is converted to a SpatialData Zarr store, described by one row of the catalog and registered in the Lamin instance [scverse/spatialdata-db](https://lamin.ai/scverse/spatialdata-db).
{{ catalog_line }} You can browse every dataset and collection on the [website]({{ site_url }}).

This Python client queries the catalog by technology, tissue, organism, gene or free text and loads the datasets you pick as `SpatialData` objects.
It builds cohorts of datasets that you can freeze, split by study and turn into training data, and it collects the citations of the studies they come from.
The `sddb-cli` command and the `sddb-mcp` server give the shell and AI agents the same queries and downloads.

```{figure} _static/landing-mosaic.webp
:alt: A grid of twelve tissue thumbnails, each labelled with its technology
:figclass: sddb-mosaic

One dataset per technology in the catalog.
```

## Installation

::::{tab-set}

:::{tab-item} pip
```bash
pip install git+https://github.com/theislab/spatialdata-db-client
```
:::

:::{tab-item} uv
```bash
uv pip install git+https://github.com/theislab/spatialdata-db-client
```
:::

::::

The client needs Python 3.12 or newer. It is not on PyPI yet, so install it from GitHub.

## Quick example

```python
from sddb import Catalog

hits = Catalog().query(technology="Visium", tissue="lung")
hits.summary()
smallest = min(hits, key=lambda ds: ds.size_bytes)
sdata = smallest.to_sdata()  # ~10 MB, copied to the local cache once
```

## Guides and reference

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` From the website
:link: start
:link-type: doc

Copy a dataset uid or a collection slug from the website and load it in Python.
:::

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` Find and load
:link: guides/find-and-load
:link-type: doc

Filter the catalog by facets or free text, look inside a store and load it.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Cohorts
:link: guides/cohorts
:link-type: doc

Combine datasets into a cohort, freeze it to exact versions and split it by study.
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Machine learning with cohorts
:link: guides/ml
:link-type: doc

Read one table across many stores and loop over the training fold of a split.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Command line and agents
:link: guides/cli-and-agents
:link-type: doc

Query, download and cite from the shell, or give an AI agent the MCP tools.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` API reference
:link: reference/api
:link-type: doc

Every public class and function of the client, with its parameters.
:::

::::

## Cite and license

The paper that describes the database is forthcoming.
Until then, cite the studies behind the data you use: `cohort.citations()` writes their BibTeX entries, and `sddb-cli cite` does the same from the shell.
Each dataset keeps the license of its source, and {doc}`cite-and-license` explains how to check and filter by license.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

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
:caption: Project

cite-and-license
changelog
```
