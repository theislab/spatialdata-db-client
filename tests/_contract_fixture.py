"""Synthetic collections for the sddb loader contract (test-only, not shipped).

:func:`write_collection` writes a small conforming collection with :func:`spatialbatch.write_collection_dataset`.
Each keyword of :func:`write_dataset` injects one defect, so a test changes exactly one thing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from scipy import ndimage, sparse

from sddb.contract import CONTRACT_VERSION, DatasetContract

N_VARS = 20
VAR_NAMES = [f"ENSG{i:011d}" for i in range(1, N_VARS + 1)]
IF_CHANNELS = ["DAPI", "CD3", "PanCK"]


def _positions(n: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Centres (y, x) of `n` cells on a regular grid over the image, and a radius that keeps them apart."""
    cols = math.ceil(math.sqrt(n * width / height))
    rows = math.ceil(n / cols)
    dy, dx = height / rows, width / cols
    ys, xs = np.meshgrid((np.arange(rows) + 0.5) * dy, (np.arange(cols) + 0.5) * dx, indexing="ij")
    return ys.ravel()[:n], xs.ravel()[:n], 0.4 * min(dy, dx)


def draw_cells(n: int, height: int, width: int, *, crescents: int = 0) -> np.ndarray:
    """Labels ``(y, x)`` uint32 with cells ``1..n`` drawn as discs.

    The first `crescents` cells are half annuli instead, whose centre of mass lies in the hole, outside the mask.
    """
    labels = np.zeros((height, width), dtype=np.uint32)
    yy, xx = np.mgrid[:height, :width]
    ys, xs, radius = _positions(n, height, width)
    for i, (cy, cx) in enumerate(zip(ys, xs, strict=True)):
        dist = np.hypot(yy - cy, xx - cx)
        if i < crescents:
            mask = (dist <= radius) & (dist >= 0.6 * radius) & (xx < cx)
        else:
            mask = dist <= radius
        labels[mask] = i + 1
    return labels


def base_contract(role: str = "he", *, uid: str = "uid0001", study_id: str = "Smith2023") -> dict[str, Any]:
    """Field values of a conforming :class:`~sddb.contract.DatasetContract` on the target grid."""
    return {
        "version": CONTRACT_VERSION,
        "uid": uid,
        "study_id": study_id,
        "catalog_release": "catalog-2026-11-01",
        "role": role,
        "grid": "target",
        "mpp_target": 0.5,
        "mpp_native": 0.25,
        "upsampled": False,
        "allow_upsample": False,
        "allow_mixed_grids": False,
        "dtype_rule": None,
        "channels": IF_CHANNELS if role == "if" else None,
        "adapters": [{"name": "Resample", "version": "0.1.0", "params": {"mpp": 0.5}}],
        "stain_reference": None,
    }


def write_dataset(
    group: zarr.Group,
    key: str,
    *,
    uid: str = "uid0001",
    study_id: str = "Smith2023",
    role: str = "he",
    height: int = 120,
    width: int = 140,
    n_cells: int = 60,
    crescents: int = 0,
    tissue: bool = True,
    image_dtype: Any = None,
    labels_shape: tuple[int, int] | None = None,
    erase_cells: int = 0,
    move_centroid: tuple[float, float] | None = None,
    outside_image: Sequence[int] | None = None,
    fractional_counts: bool = False,
    negative_counts: bool = False,
    label_ids: Sequence[int] | None = None,
    stain_reference: Mapping[str, Any] | None = None,
    var_names: Sequence[str] | None = None,
    var_columns: Mapping[str, Sequence[Any]] = MappingProxyType({}),
    zero_features: Sequence[int] = (),
    contract: Mapping[str, Any] = MappingProxyType({}),
    spatial: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> None:
    """Write one dataset at ``group[key]``; the defaults conform to the contract.

    Parameters
    ----------
    role
        ``"he"`` (uint8, 3 channels) or ``"if"`` (uint16, one channel per name in ``IF_CHANNELS``).
    crescents
        Number of cells drawn as half annuli, whose centre of mass lies outside the mask.
    tissue
        Whether to write ``labels/tissue``.
    image_dtype
        Store the image in this dtype instead of the role's.
    labels_shape
        Replace ``labels/cells`` after writing with an empty array of this shape ``(y, x)``.
    erase_cells
        Erase the pixels of this many cells from ``labels/cells``, keeping their rows.
    move_centroid
        Set the first row's ``obsm["spatial"]`` to this ``(x, y)``.
    outside_image
        Write ``obs["outside_image"]`` with these row positions True (the contract's count is not set; pass it in
        `contract`).
    fractional_counts
        Write float counts with a fractional part.
    negative_counts
        Make the first count negative.
    label_ids
        ``obs["label_id"]`` as is, instead of the cells' values ``1..n_cells``.
    stain_reference
        ``DatasetContract.stain_reference``, through the model (so it must be valid).
    var_names
        The var index, ``VAR_NAMES`` by default.
    var_columns
        Columns of var, e.g. ``feature_types`` and ``present``.
    zero_features
        Positions of features whose counts are all 0.
    contract
        Entries merged into the encoded ``uns["sddb_contract"]`` as is, so they may be invalid.
    spatial
        ``uns["spatial"][uid]`` as is, instead of the one derived from the encoded contract.
    """
    rng = np.random.default_rng(seed)
    labels = draw_cells(n_cells, height, width, crescents=crescents)
    ids = np.arange(1, n_cells + 1)
    com = np.asarray(ndimage.center_of_mass(np.ones_like(labels), labels, ids))  # (y, x)
    centres = com[:, ::-1].astype(np.float64)
    if move_centroid is not None:
        centres[0] = move_centroid
    if erase_cells:
        labels[np.isin(labels, ids[-erase_cells:])] = 0

    if role == "he":
        image = rng.integers(0, 256, (3, height, width), dtype=np.uint8)
    else:
        image = rng.integers(0, 2**16, (len(IF_CHANNELS), height, width), dtype=np.uint16)
    if image_dtype is not None:
        image = image.astype(image_dtype)

    names = list(VAR_NAMES if var_names is None else var_names)
    counts = rng.poisson(2.0, (n_cells, len(names))).astype(np.int32)
    counts[:, list(zero_features)] = 0
    if negative_counts:
        counts[0, 0] = -1
    X = sparse.csr_matrix(counts.astype(np.float32) + 0.5 if fractional_counts else counts)
    table = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "uid": uid,
                "study_id": study_id,
                "label_id": np.asarray(ids if label_ids is None else label_ids, np.int64),
            },
            index=[f"cell_{i}" for i in ids],
        ),
        var=pd.DataFrame(dict(var_columns), index=names),
    )
    if outside_image is not None:
        flags = np.zeros(n_cells, bool)
        flags[list(outside_image)] = True
        table.obs["outside_image"] = flags
    table.obsm["spatial"] = centres
    fields = base_contract(role, uid=uid, study_id=study_id)
    if stain_reference is not None:
        fields["stain_reference"] = dict(stain_reference)
    uns = DatasetContract.model_validate(fields).to_uns()
    uns.update(contract)
    table.uns["sddb_contract"] = uns
    if spatial is None:
        target = uns.get("grid") == "target"
        mpp = uns.get("mpp_target") if target else uns.get("mpp_native")
        spatial = {
            "scalefactors": {
                "mpp": mpp,
                "mpp_native": uns.get("mpp_native"),
                "native_to_written": uns["mpp_native"] / mpp if target else 1.0,
            },
            "metadata": {"role": role, "image": role, "labels": "cells", "tissue": "tissue" if tissue else None},
        }
    table.uns["spatial"] = {uid: dict(spatial)}

    label_arrays = {"cells": labels}
    if tissue:
        label_arrays["tissue"] = (labels > 0).astype(np.uint8)
    write_collection_dataset = pytest.importorskip("spatialbatch").write_collection_dataset
    write_collection_dataset(group, key=key, table=table, images={role: image}, labels=label_arrays)
    if labels_shape is not None:
        group[key]["labels"].create_array("cells", shape=labels_shape, dtype=np.uint32, fill_value=0, overwrite=True)


def write_collection(
    path: str | Path,
    n_datasets: int = 2,
    *,
    overrides: Mapping[int, Mapping[str, Any]] = MappingProxyType({}),
    **knobs: Any,
) -> Path:
    """Write a collection of `n_datasets` datasets at `path`.

    Every dataset gets `knobs` (see :func:`write_dataset`), and dataset ``i`` also ``overrides[i]``. Datasets get
    their own uid and seed.
    """
    path = Path(path)
    group = zarr.open_group(path, mode="w", zarr_format=3)
    for i in range(n_datasets):
        kwargs = {"uid": f"uid{i + 1:04d}", "seed": i, **knobs, **overrides.get(i, {})}
        write_dataset(group, f"dataset_{i}", **kwargs)
    return path
