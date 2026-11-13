"""The training-data path: adapters that bring a cohort member onto the loader contract (:mod:`sddb.contract`).

A member is opened as a :class:`LoaderItem` and runs through a chain of adapters, by default
:func:`default_chain`: :class:`Rasterize` (resample to the target mpp, cell labels), :class:`Centroids`,
:class:`TissueMask` and :class:`StainNormalize`. Each step stays lazy and chunked.

This subpackage imports squidpy, dask-image and pydantic and is not imported by ``import sddb``; install it with the
``loader`` extra.
"""

from sddb.loader._adapters import (
    ADAPTERS,
    MATERIALIZE_RECORDS,
    Centroids,
    Rasterize,
    StainNormalize,
    TissueMask,
    adapter_from_record,
    default_chain,
)
from sddb.loader._item import LoaderItem
from sddb.loader._materialize import materialize

__all__ = [
    "ADAPTERS",
    "Centroids",
    "LoaderItem",
    "MATERIALIZE_RECORDS",
    "Rasterize",
    "StainNormalize",
    "TissueMask",
    "adapter_from_record",
    "default_chain",
    "materialize",
]
