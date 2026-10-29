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
not pass validation (in the current catalog every dataset passes, so this changes nothing yet; filter on `tier` for quality), `--json` emits records instead of a table, and `--gene` (repeatable, requires all) uses the
gene index.

```
$ sddb-cli query --organism human --technology Visium --tissue lung --min-obs 5000
# catalog 2026-10-28 · 6087 datasets
uid                   technology  organism  tissue  n_obs
PntNMJy70YLUP8eq0000  Visium      human     lung    6195
gA0atNHp50LE7usk0000  Visium      human     lung    5218
B1pHBGo66lQ15oil0000  Visium      human     lung    13002
KI3j6VpkjaVwuPPY0000  Visium      human     lung    5388
```
Each match has a page on the DB site, e.g., [`PntNMJy70YLUP8eq0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-lung-pntnmjy70ylup8eq0000/).

Other options: `--assay`, `--disease`, `--tier`, `--source` (the `study_id` prefix, such as `geo`, `hest` or
`10x`), `--hne-image/--no-hne-image`, `--redistributable/--no-redistributable`, `--search`, `--min-features`,
`--min-fraction`. Case, spaces and hyphens in a value are ignored, and `--technology` also accepts the website
label, such as "10x Xenium". A value that is not in the catalog fails with the closest known value as a hint.

### facets

Without an argument it lists the facet columns, with one it lists the distinct values.

```
$ sddb-cli facets organism
ambystoma mexicanum
arabidopsis thaliana
branchiostoma floridae
crab-eating macaque
dog
glycine max
golden hamster
human
leucoraja erinaceus
macaque
mouse
pig
rat
zebra finch
zebrafish
```

### download

Copy stores by uid into a directory (resumable) and print the path of the manifest it wrote.

```
$ sddb-cli download h1okQyQN4GJd0I1l0000 --dest ./data
data/manifest.json
```

To download every member of a collection, pass its slug (the last part of the collection page URL) instead of uids:

```
$ sddb-cli download --collection <slug> --dest ./data
data/manifest.json
```

Pass uids or `--collection`, not both. `--workers` sets the number of parallel downloads. A summary line goes to stderr when the download finishes, so stdout still holds only the manifest path. The store ([`h1okQyQN4GJd0I1l0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-trunk-h1okqyqn4gjd0i1l0000/)) lands in `./data/h1okQyQN4GJd0I1l0000.zarr`.


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
(about 900 MB, downloaded once and cached), and lookups read it in batches, so a lookup needs well under 1 GB
of memory and takes about 15 seconds.

```
$ sddb-cli genes EPCAM --limit 3
uid                   fraction_obs_detected  total_counts  validation_status  tier
EVTMdgu8thDMETSm0000  1                      3.807e+04     pass               silver
S6yC0WUFUl8bFH5y0000  0.9997                 1.929e+05     pass               bronze
7nANAXEEhj6WHiCU0000  0.9997                 2.63e+05      pass               bronze
```
Datasets: [`EVTMdgu8thDMETSm0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-intestine-evtmdgu8thdmetsm0000/), [`S6yC0WUFUl8bFH5y0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-small-intestine-s6yc0wuful8bfh5y0000/), [`7nANAXEEhj6WHiCU0000`](https://spatialdata-db-web.pages.dev/datasets/visium-spatial-gene-expression-small-intestine-7nanaxeehj6whicu0000/).

`genes` defaults to `--validation all`, so datasets that failed validation are included. Pass `--validation pass` to exclude them. To restrict by tier, use `sddb-cli query --tier ...` or `Catalog().query(tier=...)`.

### cite

Write BibTeX for the studies behind the datasets that match the facets or search.

```
$ sddb-cli cite -o refs.bib --organism mouse --tissue brain --technology Visium
wrote 178 citations -> refs.bib
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
| `query` | Filter datasets by facets (including `source`, `hne_image`, `if_image` and `redistributable`), ranges (`min_obs`, `min_features`), gene (`expressing`, `min_fraction`) and free-text `search`. |
| `describe` | Catalog row and element shapes for one dataset `uid`. |
| `genes` | Datasets expressing one or more genes (`mode` `all` or `any`), optionally filtered by organism, tissue, disease or assay. Bronze datasets are included by default. |
| `features` | Datasets measuring one or more features of any modality (genes, protein targets, metabolites; labels or ids; `mode` `all` or `any`), optionally filtered by `modality`, `feature_type`, organism, tissue, disease or assay. Each row lists the typed feature hits. Bronze datasets are included by default. |
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
