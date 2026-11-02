# From the website to Python

The [DB website]({{ site_url }}) is the place to find data. This page shows what to copy from a dataset or collection page into Python, and where the same steps are in the other docs.

(install)=
## Install

```bash
pip install "spatialdata-db @ git+https://github.com/theislab/spatialdata-db-client@v0.1.3"
```

or, with uv:

```bash
uv pip install "spatialdata-db @ git+https://github.com/theislab/spatialdata-db-client@v0.1.3"
```

This installs the alpha release, v0.1.3. The `main` branch may be ahead of it, and you get the newest code if you leave out `@v0.1.3`. The package needs Python 3.12 or newer. It is not on PyPI yet, so install it from GitHub. The name `sddb` on PyPI belongs to an unrelated package.

The client keeps its cache in the user cache directory of the platform, in a folder called `sddb`. Set `SDDB_CACHE_DIR` to put it somewhere else, for example on a larger disk. The first call to `Catalog()` downloads the catalog (under 10 MB) from a GitHub release. Later calls check for a newer catalog and download it again only when one has been published.

(one-dataset)=
## One dataset

Every dataset page shows a uid in the chip at the top right. Click it to copy the full 20-character uid, then pass it to `Catalog().dataset()`. The 16-character stem that Lamin shows works as well. A stem always resolves to the latest version of the dataset, so scripts you want to reproduce should use the full uid.

```python
from sddb import Catalog

ds = Catalog().dataset("{{ small_uid }}")
print(ds.license)  # the license id of the source, e.g. CC-BY-4.0
sdata = ds.to_sdata()  # copied to the local cache on the first call, then reused
```

The uid in this example is the smallest dataset in the catalog, so it is a quick one to try. A download prints its progress on stderr, and nothing is printed when the store is already cached. Check `ds.license` before you reuse the data, and see {doc}`tutorials/04_citations_and_licences` for what the license columns mean. {doc}`tutorials/01_find_and_filter` and {doc}`tutorials/02_load_and_inspect` cover filtering and loading in more detail.

### Only some elements, in a folder of your choice

`elements` copies only the elements you name, as `"kind/name"` paths from `ds.elements()`. With `dest`, the copy goes to `dest/<uid>.zarr` instead of the cache, and `dest/manifest.json` records it as a partial copy with its element list.

```python
sdata = ds.to_sdata(elements=["tables/table"], dest="data/")
```

The client adds the shapes or labels that a kept table annotates. A second call with the same elements opens the copy without fetching anything. A later `download("data/")` of a cohort with this dataset fetches the whole store.

(one-collection)=
## One collection

The slug of a collection is the last part of its page URL, and the collection page shows it in the `slug` chip. `Catalog().collection()` takes the slug and returns a cohort with the members of the collection.

```python
from sddb import Catalog

cohort = Catalog().collection("{{ small_collection }}")
print(cohort.licenses())  # one row per license, with the number of members
sdata_by_uid = cohort.to_dict()  # {uid: SpatialData}, downloaded once into the cache
```

`licenses()` returns a table with the columns `license` and `n`. Members whose license is not known are counted under `unknown`. To keep the `.zarr` stores in a folder of your own instead of the cache, call `cohort.download("data/")`. See {doc}`tutorials/03_cohorts_and_manifests` for freezing, splitting and reproducing a cohort.

(from-uids)=
## From a list of uids

If you already know which datasets you want, for example from a paper's supplement or a previous analysis, pass their uids to `Catalog().cohort()`. Use `study_ids=` instead of `uids=` to take every dataset of the given studies. Pass exactly one of the two.

```python
from sddb import Catalog

cohort = Catalog().cohort(uids=["{{ small_uid }}"])  # add more uids to the list
print(cohort.members[["uid", "study_id"]])  # one row per dataset
sdata_by_uid = cohort.to_dict()  # {uid: SpatialData}, downloaded once into the cache
```

Add the uids you want, each copied from the chip on a dataset page. `cohort()` needs full uids, and it raises a `KeyError` that names every uid or study id that is not in the catalog, so a typo fails before anything is downloaded. See {doc}`tutorials/03_cohorts_and_manifests` for freezing, splitting and reproducing the cohort.

(reproducible)=
## Reproducible references

The catalog changes between releases. Between the releases of 2026-10-16 and 2026-10-28, 84 uids got a new version suffix and 138 stems disappeared, so a script that names data loosely can return other data a week later. Three things make a reference stable:

- `Catalog(version="2026-10-28")` reads that dated release and always gives the same rows. Without `version`, `Catalog()` reads the rolling `catalog-latest`.
- A full uid pins one version of one dataset. Its 16-character stem resolves to the latest version in the catalog you opened, so it can point at different data in a newer release.
- A `canonical_key` such as `geo:GSM9778330` is unique in the catalog, and for most datasets it equals the `study_id`. `Catalog.dataset()` accepts it. A key usually survives when a dataset is replaced by a newer version, so it is a more durable handle than a stem, but check the uid that comes back. Datasets without a `canonical_key` (null in the column) cannot be looked up this way.

```python
from sddb import Catalog

cat = Catalog(version="2026-10-28")  # a dated release, the same rows every time
ds = cat.dataset("geo:GSM9778330")  # by canonical_key; a uid or a stem works as well
print(ds.uid)  # the uid of that version, which you can record
```

Record the release date together with the uids. A `manifest.json` written from `Catalog()` without `version` has `catalog_version` set to null, so open the catalog with `version=` when the manifest should record the release. To go back to the exact data later, open the same release and pass the uids. {doc}`tutorials/03_cohorts_and_manifests` shows how to freeze a cohort and check that the catalog file has not changed.

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
: The "Technology" label and filter on the website show the catalog's `assay` column, such as "Vizgen MERSCOPE" or "10x Xenium", so to match a website filter use `Catalog().query(assay="Vizgen MERSCOPE")`. The `technology` column holds a short platform id, such as `Merscope` or `Xenium`, and `Catalog().query(technology="Merscope")` selects by it. For most platforms both columns select the same datasets. `technology` also accepts the website label, so `Catalog().query(technology="10x Xenium")` selects the `Xenium` datasets.

uid, stem and slug
: The uid is the full 20-character id of a dataset version. The stem is its first 16 characters, which Lamin uses for the latest version. The slug is the id of a collection, and the last part of its URL.

collection and cohort
: A collection is a grouping in the catalog, shown on the website. A cohort is the client object that holds a set of datasets. `Catalog.collection()` and `Catalog.query()` both return one.
