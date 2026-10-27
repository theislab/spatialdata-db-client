# Command line and agents

The package installs two commands: `sddb-cli`, a thin command line over the catalog, and `sddb-mcp`, a
[Model Context Protocol](https://modelcontextprotocol.io) server that lets an agent such as Claude query the
same catalog.

```
pip install git+https://github.com/theislab/spatialdata-db-client
```

## sddb-cli

Every command accepts `--catalog <url-or-path>` to use another catalog (default: `$SDDB_CATALOG_URL`, then the
published catalog). Errors go to stderr with exit code 1.

### query

List datasets that match facets, ranges, a gene or free text. `--validation all` includes datasets that did
not pass validation, `--json` emits records instead of a table, and `--gene` (repeatable, requires all) uses the
gene index.

```
$ sddb-cli query --organism human --technology Visium --tissue lung --min-obs 5000
# catalog 2026-10-18 · 1778 datasets
uid                   technology  organism  tissue  n_obs
PntNMJy70YLUP8eq0000  Visium      human     lung    6195
```
The match, [`PntNMJy70YLUP8eq0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-lung-pntnmjy70ylup8eq0000/), has a page on the DB site.

Other options: `--assay`, `--disease`, `--tier`, `--search`, `--min-features`, `--min-fraction`. If a value
matches nothing, the closest known value is suggested.

### facets

Without an argument it lists the facet columns, with one it lists the distinct values.

```
$ sddb-cli facets organism
arabidopsis thaliana
glycine max
human
mouse
rat
zebrafish
```

### download

Copy stores by uid into a directory (resumable) and print the path of the manifest it wrote.

```
$ sddb-cli download h1okQyQN4GJd0I1l0000 --dest ./data
data/manifest.json
```

`--workers` sets the number of parallel downloads. The store ([`h1okQyQN4GJd0I1l0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-trunk-h1okqyqn4gjd0i1l0000/)) lands in `./data/h1okQyQN4GJd0I1l0000.zarr`.


### viewer-url

Print the browser viewer link for one dataset. Nothing is downloaded.

```
$ sddb-cli viewer-url xVK6T0rajFTgOKci0000
https://vitessce.io#?edit=false&url=data:,%257B%2522version%2522...
```
Dataset: [`xVK6T0rajFTgOKci0000`](https://spatialdata-db-web.pages.dev/datasets/visium-hd-bone-element-xvk6t0rajftgokci0000/).

### genes

List datasets that express all the given genes (symbols or Ensembl ids). With one gene, datasets are ranked by
the fraction of observations in which it is detected. The first call downloads the cross-dataset gene index
(about 260 MB, cached afterwards).

```
$ sddb-cli genes EPCAM --limit 3
uid                   fraction_obs_detected  total_counts  validation_status  tier
EVTMdgu8thDMETSm0000  1                      3.807e+04     pass               silver
bOgBgMqQm9TObx470000  0.9995                 3.615e+05     pass               silver
TeLy8jHRRvoCrNMn0000  0.9989                 2.761e+05     pass               silver
```
Datasets: [`EVTMdgu8thDMETSm0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-intestine-evtmdgu8thdmetsm0000/), [`bOgBgMqQm9TObx470000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-intestine-bogbgmqqm9tobx470000/), [`TeLy8jHRRvoCrNMn0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-colon-tely8jhrrvocrnmn0000/).

`genes` defaults to `--validation all`, so datasets that failed validation are included. Pass `--validation pass` to exclude them. To restrict by tier, use `sddb-cli query --tier ...` or `Catalog().query(tier=...)`.

### cite

Write BibTeX for the studies behind the datasets that match the facets or search.

```
$ sddb-cli cite -o refs.bib --organism mouse --tissue brain --technology Visium
wrote 55 citations -> refs.bib
```

Studies without a matching citation key are skipped, and the command exits with an error if no datasets match at all.

## sddb-mcp

The MCP server speaks stdio and exposes the catalog to an agent. Install the extra:

```
pip install "spatialdata-db[mcp] @ git+https://github.com/theislab/spatialdata-db-client"
```

### Claude Desktop

Add the server to `claude_desktop_config.json` and restart the app:

```json
{
  "mcpServers": {
    "spatialdata-db": {
      "command": "sddb-mcp"
    }
  }
}
```

### Claude Code

Register it once from a terminal:

```
claude mcp add spatialdata-db -- sddb-mcp
```

or commit the same JSON as `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "spatialdata-db": {
      "command": "sddb-mcp"
    }
  }
}
```

If `sddb-mcp` is not on the `PATH` of the app, use its absolute path as `command`.

### Tools

| Tool | What it does |
|---|---|
| `query` | Filter datasets by facets, ranges (`min_obs`, `min_features`), gene (`expressing`, `min_fraction`) and free-text `search`. |
| `describe` | Catalog row and element shapes for one dataset `uid`. |
| `genes` | Datasets expressing one or more genes (`mode` `all` or `any`), optionally filtered by organism, tissue, disease or assay. Bronze datasets are included by default. |
| `facets` | List the facet columns, or the distinct values of one field. |
| `cite` | BibTeX for the studies of a filtered cohort. |
| `download` | Plan a download (per-store and total sizes) or, with `download=true`, fetch the cohort into the server download directory. |

Two limits keep an agent safe:

- `query` returns at most `MAX_ROWS = 200` rows. When there are more, the last item is a
  `{"note": "truncated: ..."}` entry telling the agent to narrow the query.
- `download` never writes where the agent asks. Files go to the directory in the `SDDB_MCP_DOWNLOAD_DIR`
  environment variable, or to a `mcp-downloads` folder in the user cache directory. By default it only plans, so
  the agent can report sizes before anything is copied.

Set the variable in the server entry:

```json
{
  "mcpServers": {
    "spatialdata-db": {
      "command": "sddb-mcp",
      "env": { "SDDB_MCP_DOWNLOAD_DIR": "/data/sddb-downloads" }
    }
  }
}
```
