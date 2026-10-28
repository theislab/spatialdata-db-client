# From the website to Python

The [DB website]({{ site_url }}) is the place to find data. This page shows what to copy from a dataset or collection page into Python, and where the same steps are in the other docs.

(install)=
## Install

```bash
pip install git+https://github.com/theislab/spatialdata-db-client
```

or, with uv:

```bash
uv pip install git+https://github.com/theislab/spatialdata-db-client
```

The package needs Python 3.12 or newer. It is not on PyPI yet, so install it from GitHub. The name `sddb` on PyPI belongs to an unrelated package.

The client keeps its cache in the user cache directory of the platform, in a folder called `sddb`. Set `SDDB_CACHE_DIR` to put it somewhere else, for example on a larger disk. The first call to `Catalog()` downloads the catalog (about 7 MB) from a GitHub release. Later calls check for a newer catalog and download it again only when one has been published.

(one-dataset)=
## One dataset

Every dataset page shows a uid in the chip at the top right. Click it to copy the full 20-character uid, then pass it to `Catalog().dataset()`. The 16-character stem that Lamin shows works as well.

```python
from sddb import Catalog

ds = Catalog().dataset("{{ small_uid }}")
print(ds.license)  # the license id of the source, e.g. CC-BY-4.0
sdata = ds.to_sdata()  # copied to the local cache on the first call, then reused
```

The uid in this example is the smallest dataset in the catalog, so it is a quick one to try. A download prints its progress on stderr, and nothing is printed when the store is already cached. Check `ds.license` before you reuse the data, and see {doc}`cite-and-license` for what the license columns mean. {doc}`guides/find-and-load` covers filtering and loading in more detail.

(one-collection)=
## One collection

The slug of a collection is the last part of its page URL, and the collection page shows it in the `slug` chip. `Catalog().collection()` takes the slug and returns a cohort with the members of the collection.

```python
from sddb import Catalog

cohort = Catalog().collection("{{ small_collection }}")
print(cohort.licenses())  # one row per license, with the number of members
sdata_by_uid = cohort.to_dict()  # {uid: SpatialData}, downloaded once into the cache
```

`licenses()` returns a table with the columns `license` and `n`. Members whose license is not known are counted under `unknown`. To keep the `.zarr` stores in a folder of your own instead of the cache, call `cohort.download("data/")`. See {doc}`guides/cohorts` for freezing, splitting and reproducing a cohort.

(cli)=
## Command line

The same two steps work from the shell:

```bash
sddb-cli download <uid> --dest data/
sddb-cli download --collection <slug> --dest data/
```

Both print the path of the manifest they wrote on stdout and a one-line summary on stderr. {doc}`guides/cli-and-agents` lists the other commands and the MCP server for agents.

(lamin)=
## Lamin

The data also lives in the Lamin database `scverse/spatialdata-db`. The Lamin tab on the website shows these lines:

```python
import lamindb as ln

db = ln.DB("scverse/spatialdata-db")
artifact = db.Artifact.get("{{ small_stem }}")  # the 16-character stem of the uid
sdata = artifact.load()

coll = db.Collection.get(key="<Lamin key>")  # a collection, by its Lamin key
```

The Lamin key of a collection is shown in the Lamin tab on its page. It often equals the slug, but not always, so copy it from the tab.

This route needs `pip install lamindb` and network access to the Lamin database. Access is limited during the alpha, so the calls can fail or hang outside an allowed network. If they do, use the client as shown above.

(words)=
## Words used on both sides

technology and assay
: The technology is the platform family, such as Visium or Xenium. The assay is the product string within it, such as "Visium Spatial Gene Expression". Both are columns of the catalog.

uid, stem and slug
: The uid is the full 20-character id of a dataset version. The stem is its first 16 characters, which Lamin uses for the latest version. The slug is the id of a collection, and the last part of its URL.

collection and cohort
: A collection is a grouping in the catalog, shown on the website. A cohort is the client object that holds a set of datasets. `Catalog.collection()` and `Catalog.query()` both return one.
