"""Lean read client for the spatialdata-db spatial-omics collection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatialdata-db")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from sddb.adapter import TableAdapter, TaskAdapter
from sddb.catalog import Catalog
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset
from sddb.geometry import table_centroids
from sddb.manifest import CatalogChangedError, CohortDriftError, CohortManifest, Manifest, ManifestVersionMismatch
from sddb.remote import open_sdata
from sddb.roles import resolve
from sddb.split import SplitManifest

__all__ = [
    "Catalog",
    "CatalogChangedError",
    "CohortDriftError",
    "CohortManifest",
    "Dataset",
    "Manifest",
    "ManifestVersionMismatch",
    "SpatialDataCohort",
    "SplitManifest",
    "TableAdapter",
    "TaskAdapter",
    "__version__",
    "open_sdata",
    "resolve",
    "table_centroids",
]
