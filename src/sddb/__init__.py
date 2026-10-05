"""Lean read client for the spatialdata-db spatial-omics collection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatialdata-db")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
