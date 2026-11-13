"""The loader contract (version 0.1.0): what a cohort materialised for :mod:`spatialbatch` looks like on disk.

A collection is a :class:`spatialbatch.SpatialDatasetCollection`: a zarr v3 group with one ``dataset_<i>`` group per
cohort member, each written by :func:`spatialbatch.write_collection_dataset`. Per dataset:

- the table (annbatch layout), rows Morton-sorted on ``obsm["spatial"]``, each cell's centre ``(x, y)`` float64 in
  pixel-index coordinates of the written grid (pixel ``j`` has its centre at ``j``), inside ``[0, w) x [0, h)``. It is
  the centre of mass of the cell's label in ``labels/cells``, except for the ``n_centroid_fallback`` cells that have
  no pixel there (smaller than a pixel of the written grid), whose centre comes from the source (see
  :class:`DatasetContract`);
- ``obs["uid"]`` and ``obs["study_id"]`` of the source member, and ``obs["label_id"]``, the integer value of the
  row's cell in ``labels/cells``;
- ``X`` integer counts (an integer dtype, or a float dtype with integral values) over one var index, the same in
  every dataset: the union of the members' features. ``var["feature_types"]`` comes from the source, and the
  ``"Gene Expression"`` rows are Ensembl ids (``^ENS``). ``var["present"]`` says which features this dataset's
  source measured; a feature it did not measure is all 0 in ``X``. ``n_features_present`` and
  ``n_features_filled`` in the contract count the two;
- ``images/<role>``, the one image ``(c, y, x)``: ``he`` uint8 with 3 channels or ``if`` uint16 with one channel per
  name in ``channels``. One role per collection;
- ``labels/cells`` ``(y, x)`` integer on the image's grid, and optionally ``labels/tissue``. Images and labels are
  plain arrays, the identity onto the written grid: no ``coordinateTransformations`` or ``multiscales`` attrs;
- ``uns["sddb_contract"]``, a :class:`DatasetContract`;
- ``uns["spatial"][uid]``, a :class:`SpatialEntry`. It follows Squidpy's layout (``scalefactors``, ``metadata``) but
  holds no image arrays: the image is ``images/<role>`` of the dataset.

Every sampled cell (``sample`` cells drawn with ``seed``) must have at least one pixel of its ``label_id``, except for
at most ``n_centroid_fallback`` of them, and at least 99% of the centroids of those with pixels, rounded to the
pixel, must lie inside their own label: a concave cell's centre of mass can lie outside its mask.

Resolution: on the ``target`` grid the image was resampled to ``mpp_target``. A target finer than the native grid
(``mpp_native / mpp_target > 1``, by any factor) is upsampling, which needs ``allow_upsample``. A dataset whose mpp is
unknown keeps its ``native`` grid; a collection mixes the two only if every dataset sets ``allow_mixed_grids``.

This subpackage imports pydantic and is not imported by ``import sddb``; install it with the ``loader`` extra.
"""

from sddb.contract._models import (
    CONTRACT_VERSION,
    AdapterRecord,
    DatasetContract,
    Scalefactors,
    SpatialEntry,
    SpatialMetadata,
)
from sddb.contract._validate import ContractError, assert_valid, validate, validate_dataset

__all__ = [
    "CONTRACT_VERSION",
    "AdapterRecord",
    "ContractError",
    "DatasetContract",
    "Scalefactors",
    "SpatialEntry",
    "SpatialMetadata",
    "assert_valid",
    "validate",
    "validate_dataset",
]
