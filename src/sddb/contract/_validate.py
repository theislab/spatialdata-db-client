"""Checks of a written collection against the contract: the ``uns`` models plus the zarr arrays."""

from __future__ import annotations

import json
import math
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from anndata.io import read_elem, sparse_dataset
from pydantic import ValidationError

from sddb.contract._models import CONTRACT_VERSION, DatasetContract, SpatialEntry

_DATASET = re.compile(r"^dataset_(\d+)$")
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.+-]+)?$")
_DTYPES = {"he": np.dtype(np.uint8), "if": np.dtype(np.uint16)}
_LABELS = {"cells", "tissue"}
_TRANSFORM_ATTRS = {"coordinateTransformations", "multiscales", "transform", "transformations"}
_INSIDE = 0.99  # share of sampled centroids that must lie inside their own label
_WINDOW = 64  # half side of the window searched around a centroid before scanning the whole labels array


class ContractError(ValueError):
    """A collection does not conform to the contract; `issues` lists why."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)
        super().__init__("collection does not conform to the sddb contract:\n- " + "\n- ".join(self.issues))


@dataclass
class _Result:
    issues: list[str] = field(default_factory=list)
    contract: DatasetContract | None = None
    var: pd.Index | None = None


def validate_dataset(group: zarr.Group, *, sample: int = 2000, seed: int = 0) -> list[str]:
    """Check one dataset of a collection against the contract.

    Parameters
    ----------
    group
        The dataset's group, ``collection["dataset_<i>"]``, as written by :func:`spatialbatch.write_collection_dataset`.
    sample
        Number of cells (at most all of them) drawn for the checks that read pixels or counts per cell.
    seed
        Seed of the draw, so that a rerun checks the same cells.

    Returns
    -------
    The issues found; an empty list means the dataset conforms. Malformed input is reported as an issue, not raised.
    """
    return _validate_dataset(group, sample=sample, seed=seed).issues


def validate(path: str | Path, *, sample: int = 2000, seed: int = 0) -> list[str]:
    """Check a collection: every dataset (see :func:`validate_dataset`) plus the checks across datasets.

    Across datasets, the collection has one image role, the same var index in every dataset, one ``mpp_target`` on
    the target grid, and grids of one kind (``target`` or ``native``) unless every dataset sets ``allow_mixed_grids``.

    Returns
    -------
    The issues found, each prefixed by its dataset (``dataset_<i>:``) or by ``collection:``; an empty list means the
    collection conforms.
    """
    try:
        root = zarr.open_group(Path(path), mode="r")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        return [f"collection: cannot open {path} as a zarr group: {e!r}"]
    keys = sorted((k for k in root.group_keys() if _DATASET.match(k)), key=lambda k: int(k.split("_")[1]))
    if not keys:
        return [f"collection: {path} has no dataset_<i> group"]
    issues: list[str] = []
    results: dict[str, _Result] = {}
    for key in keys:
        results[key] = result = _validate_dataset(root[key], sample=sample, seed=seed)
        issues += [f"{key}: {issue}" for issue in result.issues]

    contracts = {k: r.contract for k, r in results.items() if r.contract is not None}
    if len(roles := {c.role for c in contracts.values()}) > 1:
        issues.append(f"collection: one image role per collection, got {sorted(roles)}")
    targets = {k: c.mpp_target for k, c in contracts.items() if c.grid == "target"}
    if len(targets) > 1:
        first_key, first_mpp = next(iter(targets.items()))
        if differ := [k for k, mpp in targets.items() if not _same(mpp, first_mpp)]:
            issues.append(
                f"collection: every target-grid dataset has one mpp_target, {first_key} has {first_mpp} but "
                f"{', '.join(f'{k} has {targets[k]}' for k in differ)}"
            )
    if len({c.grid for c in contracts.values()}) > 1:
        unflagged = [k for k, c in contracts.items() if not c.allow_mixed_grids]
        if unflagged:
            issues.append(
                "collection: mixes target and native grids, which needs allow_mixed_grids on every dataset; "
                f"not set on {unflagged}"
            )
    vars_ = {k: r.var for k, r in results.items() if r.var is not None}
    if vars_:
        first, ref = next(iter(vars_.items()))
        for key, var in vars_.items():
            if not var.equals(ref):
                issues.append(f"{key}: var index differs from {first}'s ({_index_diff(var, ref)})")
    return issues


def assert_valid(path: str | Path, *, sample: int = 2000, seed: int = 0) -> None:
    """Raise :class:`ContractError` if :func:`validate` finds issues."""
    if issues := validate(path, sample=sample, seed=seed):
        raise ContractError(issues)


def _validate_dataset(group: Any, *, sample: int, seed: int) -> _Result:
    out = _Result()
    # Malformed input is an issue, never an exception; the checks of this dataset stop at the first such error.
    try:
        _check_dataset(group, out, sample=sample, seed=seed)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, IndexError, AttributeError, OverflowError) as e:
        # name where it was raised, so a bug in these checks is not mistaken for a bad dataset
        frame = traceback.extract_tb(e.__traceback__)[-1]
        out.issues.append(
            f"not an annbatch table or contract unreadable: {e!r} (at {Path(frame.filename).name}:{frame.lineno})"
        )
    return out


def _check_dataset(group: Any, out: _Result, *, sample: int, seed: int) -> None:
    try:
        contract = DatasetContract.from_uns(_read(group["uns"]["sddb_contract"]))
    except KeyError:
        out.issues.append("the table has no uns['sddb_contract']")
        return
    except ValidationError as e:
        out.issues += [_pydantic_issue("uns['sddb_contract']", err) for err in e.errors()]
        return
    out.contract = contract
    out.issues += _check_contract(contract)
    try:
        entry = SpatialEntry.from_uns(_read(group["uns"]["spatial"][contract.uid]))
    except KeyError:
        out.issues.append(f"the table has no uns['spatial'][{contract.uid!r}]")
    except ValidationError as e:
        out.issues += [_pydantic_issue(f"uns['spatial'][{contract.uid!r}]", err) for err in e.errors()]
    else:
        out.issues += _check_spatial(contract, entry)

    out.issues += _check_encoding(group)
    issues, image, cells = _check_elements(group, contract)
    out.issues += issues
    n_obs = len(_read(group["obs"][group["obs"].attrs["_index"]]))
    issues, label_id = _check_obs(group, contract)
    out.issues += issues
    if label_id is not None and cells is not None:
        issues, label_id = _check_label_dtype(label_id, cells.dtype)
        out.issues += issues
    issues, centres = _check_centres(group, n_obs, None if image is None else image.shape[-2:])
    out.issues += issues

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n_obs, size=min(sample, n_obs), replace=False))
    if cells is not None and centres is not None and label_id is not None and len(rows):
        out.issues += _check_cells(cells, centres[rows], label_id[rows])
    out.issues += _check_counts(group, rows)
    issues, out.var = _check_var(group)
    out.issues += issues


def _read(node: Any) -> Any:
    """:func:`anndata.io.read_elem`, typed loosely: what it returns depends on the element's encoding."""
    return read_elem(node)


def _pydantic_issue(where: str, err: Any) -> str:
    loc = ".".join(str(part) for part in err["loc"])
    return f"{where}: {loc + ': ' if loc else ''}{err['msg']}"


def _check_contract(contract: DatasetContract) -> list[str]:
    """Checks of the contract fields that the model alone cannot make: the version and the upsampling."""
    issues = []
    if not (m := _SEMVER.match(contract.version)):
        issues.append(f"contract version {contract.version!r} is not semver")
    elif m.groups()[:2] != tuple(CONTRACT_VERSION.split(".")[:2]):
        issues.append(f"contract version {contract.version} is not compatible with {CONTRACT_VERSION}")
    if contract.grid == "native" and contract.upsampled:
        issues.append("upsampled is set on a native grid, which is not resampled")
    if contract.grid == "target" and contract.mpp_native is not None and contract.mpp_target is not None:
        # Both are set on a target grid (the model requires them); the test only narrows the types.
        factor = contract.mpp_native / contract.mpp_target
        if factor > 1 and not contract.upsampled:
            issues.append(
                f"the target grid (mpp {contract.mpp_target}) is finer than the native one (mpp {contract.mpp_native}): "
                f"a factor of {factor:g} > 1 is upsampling (there is no tolerance), but upsampled is False"
            )
        if factor <= 1 and contract.upsampled:
            issues.append(f"upsampled is True, but the resampling factor {factor:g} is not above 1")
    return issues


def _check_spatial(contract: DatasetContract, entry: SpatialEntry) -> list[str]:
    """``uns["spatial"][uid]`` agrees with the contract and names the fixed elements."""
    where = f"uns['spatial'][{contract.uid!r}]"
    meta, scale = entry.metadata, entry.scalefactors
    issues = []
    if meta.role != contract.role:
        issues.append(f"{where}: metadata.role is {meta.role!r}, the contract's role is {contract.role!r}")
    if meta.image != contract.role:
        issues.append(f"{where}: metadata.image is {meta.image!r}, the image is named after its role {contract.role!r}")
    if meta.tissue not in (None, "tissue"):
        issues.append(f"{where}: metadata.tissue is {meta.tissue!r}, the tissue mask is named 'tissue'")
    written = contract.mpp_target if contract.grid == "target" else contract.mpp_native
    if not _same(scale.mpp, written):
        issues.append(f"{where}: scalefactors.mpp is {scale.mpp}, the written grid has mpp {written}")
    if not _same(scale.mpp_native, contract.mpp_native):
        issues.append(
            f"{where}: scalefactors.mpp_native is {scale.mpp_native}, the contract's is {contract.mpp_native}"
        )
    if contract.grid == "native":
        expected: float | None = 1.0
    elif contract.mpp_native is not None and contract.mpp_target is not None:
        expected = contract.mpp_native / contract.mpp_target
    else:  # pragma: no cover - the model requires both mpps on a target grid
        expected = None
    if expected is not None and not _same(scale.native_to_written, expected):
        issues.append(f"{where}: scalefactors.native_to_written is {scale.native_to_written}, expected {expected:g}")
    return issues


def _same(a: float | None, b: float | None) -> bool:
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9)


def _check_encoding(group: Any) -> list[str]:
    meta = group.attrs.get("spatialbatch")
    if not isinstance(meta, dict) or meta.get("centres") != "obsm/spatial":
        return ["not written by spatialbatch.write_collection_dataset (no attrs['spatialbatch'] with centres in obsm)"]
    return []


def _check_elements(group: Any, contract: DatasetContract) -> tuple[list[str], Any, Any]:
    """The image ``images/<role>`` and the labels ``labels/cells`` (and optional ``labels/tissue``) on one grid.

    Returns the issues, the image and the cells labels; an element is ``None`` if it cannot be checked further.
    """
    issues: list[str] = []
    role = contract.role
    images = group.get("images")
    names = sorted(images.keys()) if isinstance(images, zarr.Group) else []
    if names != [role]:
        issues.append(f"images must be exactly ['{role}'] for role {role!r}, got {names}")
    image = images.get(role) if isinstance(images, zarr.Group) else None
    if image is not None and not isinstance(image, zarr.Array):
        issues.append(f"images/{role} must be one array, not a group (no multiscales)")
        image = None
    if image is not None:
        dtype = _DTYPES[role]
        if image.dtype != dtype:
            issues.append(f"images/{role} has dtype {image.dtype}, role {role!r} needs {dtype}")
        n_channels = 3 if role == "he" else len(contract.channels or [])
        if image.ndim != 3 or image.shape[0] != n_channels:
            issues.append(f"images/{role} has shape {image.shape}, role {role!r} needs ({n_channels}, y, x)")
            image = None if image.ndim != 3 else image

    labels = group.get("labels")
    present = sorted(labels.keys()) if isinstance(labels, zarr.Group) else []
    if "cells" not in present:
        issues.append("there is no labels/cells")
    if extra := sorted(set(present) - _LABELS):
        issues.append(f"labels may only be {sorted(_LABELS)}, got also {extra}")
    cells = None
    for name in sorted(set(present) & _LABELS):
        array = labels[name]
        if not isinstance(array, zarr.Array):
            issues.append(f"labels/{name} must be one array, not a group (no multiscales)")
            continue
        ok = True
        if array.ndim != 2:
            issues.append(f"labels/{name} has shape {array.shape}, needs (y, x)")
            ok = False
        if not (np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_):
            issues.append(f"labels/{name} has dtype {array.dtype}, needs an integer dtype")
            ok = False
        if image is not None and array.shape != image.shape[-2:]:
            issues.append(f"labels/{name} has shape {array.shape}, the image's (y, x) is {image.shape[-2:]}")
            ok = False
        if name == "cells" and ok:
            cells = array

    # The arrays are the written grid: any transform or multiscales attr would place them elsewhere.
    paths = [kind for kind, node in (("images", images), ("labels", labels)) if isinstance(node, zarr.Group)]
    for path in (*paths, *(f"images/{n}" for n in names), *(f"labels/{n}" for n in present)):
        if found := sorted(_TRANSFORM_ATTRS & set(group[path].attrs.keys())):
            issues.append(f"{path} has {found} attrs; elements must be the identity onto the written grid")
    return issues, image, cells


def _check_obs(group: Any, contract: DatasetContract) -> tuple[list[str], np.ndarray | None]:
    """Obs has ``uid`` and ``study_id`` of the contract and an integer ``label_id``; returns the label ids."""
    obs = group["obs"]
    columns = list(obs.attrs.get("column-order", []))
    issues = [f"obs has no column {c!r}" for c in ("uid", "study_id", "label_id") if c not in columns]
    for col in ("uid", "study_id"):
        if col in columns and (bad := {str(v) for v in _read(obs[col])} - {getattr(contract, col)}):
            issues.append(f"obs[{col!r}] holds {sorted(bad)[:3]}, the contract's {col} is {getattr(contract, col)!r}")
    if "label_id" not in columns:
        return issues, None
    label_id = np.asarray(_read(obs["label_id"]))
    if not np.issubdtype(label_id.dtype, np.integer):
        issues.append(f"obs['label_id'] has dtype {label_id.dtype}, needs an integer dtype")
        return issues, None
    # 0 is the background of labels/cells: a row with label_id 0 on background would pass R2.
    if (label_id <= 0).any():
        issues.append(
            f"obs['label_id'] has {int((label_id <= 0).sum())} values <= 0, needs labels > 0 (0 is background)"
        )
        return issues, None
    values, counts = np.unique(label_id, return_counts=True)
    if (counts > 1).any():
        issues.append(f"obs['label_id'] is not unique, e.g. {values[counts > 1][:5].tolist()} repeat")
        return issues, None
    return issues, label_id


def _check_label_dtype(label_id: np.ndarray, dtype: np.dtype[Any]) -> tuple[list[str], np.ndarray | None]:
    """Every ``label_id`` is a value that ``labels/cells`` can hold; returns the ids if so."""
    high = 1 if dtype == np.bool_ else np.iinfo(dtype).max
    if (label_id > high).any():
        return [f"obs['label_id'] has values up to {int(label_id.max())}, above the {dtype} max of labels/cells"], None
    return [], label_id


def _check_centres(group: Any, n_obs: int, grid: tuple[int, ...] | None) -> tuple[list[str], np.ndarray | None]:
    """``obsm["spatial"]`` is ``(n_obs, 2)`` finite ``(x, y)`` inside ``[0, w) x [0, h)``; returns it if so."""
    try:
        centres = np.asarray(group["obsm"]["spatial"][...], dtype=np.float64)
    except KeyError:
        return ["there is no obsm['spatial']"], None
    if centres.shape != (n_obs, 2):
        return [f"obsm['spatial'] has shape {centres.shape}, needs ({n_obs}, 2)"], None
    if not np.isfinite(centres).all():
        return [f"obsm['spatial'] has {int((~np.isfinite(centres)).any(axis=1).sum())} rows that are not finite"], None
    if grid is None:
        return [], None
    height, width = grid
    x, y = centres[:, 0], centres[:, 1]
    outside = (x < 0) | (x >= width) | (y < 0) | (y >= height)
    if outside.any():
        first = centres[np.argmax(outside)].tolist()
        return [
            f"obsm['spatial'] has {int(outside.sum())} centres (x, y) outside the image [0, {width}) x [0, {height}), "
            f"e.g. {first}"
        ], None
    return [], centres


def _check_cells(cells: zarr.Array[Any], centres: np.ndarray, label_id: np.ndarray) -> list[str]:
    """R2 on the sampled cells: each has pixels in ``labels/cells``, and >= 99% of centroids lie in their own label."""
    height, width = cells.shape
    xs = np.clip(np.floor(centres[:, 0] + 0.5).astype(np.int64), 0, width - 1)
    ys = np.clip(np.floor(centres[:, 1] + 0.5).astype(np.int64), 0, height - 1)
    inside = np.asarray(cells.get_coordinate_selection((ys, xs))) == label_id
    issues = []
    if absent := _absent(cells, label_id[~inside], ys[~inside], xs[~inside]):
        issues.append(
            f"{len(absent)} of {len(label_id)} sampled cells have no pixel of their obs['label_id'] in labels/cells, "
            f"e.g. label_id {sorted(absent)[:5]}"
        )
    if (share := inside.mean()) < _INSIDE:
        issues.append(
            f"{share:.1%} of {len(label_id)} sampled centroids (rounded to the pixel) lie inside their own label in "
            f"labels/cells, fewer than {_INSIDE:.0%}. The 1% tolerance is there because a concave cell's centre of "
            "mass can lie outside its mask; more than that means the centroids and the labels disagree"
        )
    return issues


def _absent(cells: zarr.Array[Any], ids: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> set[int]:
    """The `ids` with no pixel in `cells`: first searched near their centroid, then over the whole array by block."""
    missing = set()
    for value, y, x in zip(ids.tolist(), ys.tolist(), xs.tolist(), strict=True):
        window = cells[max(y - _WINDOW, 0) : y + _WINDOW, max(x - _WINDOW, 0) : x + _WINDOW]
        if not (np.asarray(window) == value).any():
            missing.add(value)
    if not missing:
        return missing
    height, width = cells.shape
    step = max((cells.shards or cells.chunks)[0], 1024)
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            block = np.asarray(cells[y0 : y0 + step, x0 : x0 + step])
            missing -= set(np.intersect1d(block, np.fromiter(missing, dtype=block.dtype)).tolist())
            if not missing:
                return missing
    return missing


def _check_counts(group: Any, rows: np.ndarray) -> list[str]:
    """X holds counts: an integer or float dtype whose sampled values are non-negative and, if float, integral."""
    X = group["X"]
    sparse = isinstance(X, zarr.Group)
    dtype = X["data"].dtype if sparse else X.dtype
    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
        return [f"X has dtype {dtype}, needs integer counts"]
    if not len(rows):
        return []
    if sparse:
        csr: Any = sparse_dataset(X)[rows]
        values = csr.data
    else:
        values = np.asarray(X.get_orthogonal_selection((rows, slice(None))))
    issues = []
    if np.issubdtype(dtype, np.floating) and not (np.isfinite(values).all() and (values == np.round(values)).all()):
        issues.append(f"X has dtype {dtype} with non-integral values in the sampled rows, needs integer counts")
    if (values < 0).any():
        issues.append(f"X has {int((values < 0).sum())} negative values in the sampled rows, needs counts >= 0")
    return issues


def _check_var(group: Any) -> tuple[list[str], pd.Index | None]:
    """The var index is Ensembl gene ids; returns it for the comparison across datasets."""
    var = group["var"]
    index = pd.Index(np.asarray(_read(var[var.attrs["_index"]])).astype(str))
    bad = index[~index.str.match(r"^ENS")]
    if len(bad):
        return [f"var index has {len(bad)} names not matching ^ENS (Ensembl ids), e.g. {list(bad[:3])}"], index
    return [], index


def _index_diff(var: pd.Index, ref: pd.Index) -> str:
    if len(var) != len(ref):
        return f"{len(var)} names vs {len(ref)}"
    i = int(np.argmax(var.to_numpy() != ref.to_numpy()))
    return f"first difference at position {i}: {var[i]!r} vs {ref[i]!r}"
