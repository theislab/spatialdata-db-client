"""Lean read client for the spatialdata-db spatial-omics collection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatialdata-db")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from sddb.catalog import Catalog
from sddb.dataset import Dataset, Results
from sddb.remote import open_sdata

__all__ = ["Catalog", "Dataset", "Results", "__version__", "open_sdata"]
