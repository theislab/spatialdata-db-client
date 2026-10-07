"""Lean read client for the spatialdata-db spatial-omics collection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatialdata-db")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from sddb.adapter import PreflightReport, TaskAdapter
from sddb.catalog import Catalog
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset
from sddb.freeze import CohortDriftError, FrozenCohort
from sddb.manifest import Manifest, ManifestVersionMismatch
from sddb.remote import open_sdata
from sddb.split import SplitManifest

__all__ = [
    "Catalog",
    "CohortDriftError",
    "Dataset",
    "FrozenCohort",
    "Manifest",
    "ManifestVersionMismatch",
    "PreflightReport",
    "SpatialDataCohort",
    "SplitManifest",
    "TaskAdapter",
    "__version__",
    "open_sdata",
]
