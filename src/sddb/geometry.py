"""Geometry helpers that relate table rows to image pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spatialdata import SpatialData


def table_centroids(sdata: SpatialData, table: str, image: str) -> np.ndarray:
    """Centroid of each table row in an image's scale-0 pixel space.

    The table must annotate exactly one region element. The region's centroids are taken in a coordinate system shared
    with the image and mapped through the inverse of the image's transformation to that system, so no particular
    transform (e.g. identity) is assumed.

    Parameters
    ----------
    sdata
        The SpatialData object holding the table, its region element and the image.
    table
        Name of the table in ``sdata.tables``.
    image
        Name of the image in ``sdata.images``; multiscale images use their scale-0 pixel grid.

    Returns
    -------
    ``(n_obs, 2)`` float64 array of ``(x, y)`` pixel coordinates, rows aligned to ``sdata.tables[table].obs``.

    Raises
    ------
    ValueError
        If the table annotates more than one region, or the region and image share no coordinate system.
    """
    from spatialdata import get_centroids
    from spatialdata.models import get_table_keys
    from spatialdata.transformations import get_transformation

    adata = sdata.tables[table]
    region, _, instance_key = get_table_keys(adata)
    regions = [region] if isinstance(region, str) else list(region)
    if len(regions) != 1:
        raise ValueError(f"table {table!r} annotates {len(regions)} regions; exactly one region is required")
    region_name = regions[0]

    region_el = sdata[region_name]
    image_el = sdata.images[image]
    region_cs = set(get_transformation(region_el, get_all=True))
    image_tf = get_transformation(image_el, get_all=True)
    shared = sorted(region_cs & set(image_tf))
    if not shared:
        raise ValueError(f"{region_name!r} and image {image!r} share no coordinate system")
    cs = shared[0]

    cents = get_centroids(region_el, coordinate_system=cs).compute()
    xy = cents[["x", "y"]].to_numpy(dtype=np.float64)
    affine = image_tf[cs].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    inv = np.linalg.inv(affine)
    pix = xy @ inv[:2, :2].T + inv[:2, 2]

    pos = cents.index.get_indexer(adata.obs[instance_key])
    if (pos < 0).any():
        raise ValueError(f"table {table!r} has {instance_key!r} values missing from {region_name!r}")
    return np.ascontiguousarray(pix[pos])
