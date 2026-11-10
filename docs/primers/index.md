# Primers

A primer is the analysis behind one of the website's collection pages, for example the Atera collection. Each
primer is a notebook that loads its datasets from a pinned catalog release, checks their quality and integrates
them. The website shows the main charts and links here for the full analysis.

The notebooks are shown with the outputs of their last run. They were executed on a GPU node of our cluster, each
in its own environment (`pixi.toml` and `pixi.lock` next to the notebook), and the docs build does not run them.
To rerun one, see `docs/primers/run_primer.sh`.

```{toctree}
:maxdepth: 1
:glob:

*/primer
```
