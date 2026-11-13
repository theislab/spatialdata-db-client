"""Write a cohort as a loader collection (:mod:`sddb.contract`), in process: the ``runner="local"`` path."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from anndata.io import read_elem
from spatialbatch import write_collection_dataset  # type: ignore[import-untyped]

from sddb.concat import _harmonize_var
from sddb.contract import (
    CONTRACT_VERSION,
    AdapterRecord,
    ContractError,
    DatasetContract,
    SpatialEntry,
    assert_valid,
    validate_dataset,
)
from sddb.loader._adapters import GENE_KEY as _GENE_KEY
from sddb.loader._adapters import Rasterize, StainNormalize
from sddb.loader._item import LoaderItem

if TYPE_CHECKING:
    from anndata import AnnData

    from sddb.adapter import TaskAdapter

logger = logging.getLogger(__name__)

_DATASET = re.compile(r"^dataset_\d+$")
_GENE_EXPRESSION = "Gene Expression"
_GENE_KEY_VERSION = "0.1.0"


def materialize(
    cohort: Iterable[Any],
    dest: str | Path,
    *,
    adapters: Sequence[TaskAdapter],
    role: Literal["he", "if"],
    roles: Mapping[str, str] | None = None,
    mpp_native: Mapping[str, float] | None = None,
    obs_columns: Sequence[str] = (),
    runner: Literal["local", "snakemake"] = "local",
    executor: Literal["local", "slurm"] = "local",
    workers: int = 1,
    allow_mixed_grids: bool = False,
) -> Path:
    """Write every member of `cohort` as one dataset of a loader collection at `dest`, and validate it.

    Members are opened lazily (``to_sdata``) and written in uid order as ``dataset_0``, ``dataset_1``, ... The run
    state lives beside the collection, in ``<dest>.sddb/`` (``done/<uid>`` markers and ``scratch/``), so that `dest`
    holds only zarr groups. Per member:

    1. its table is re-keyed to Ensembl ids (``gene_key="ensembl"`` of :meth:`~sddb.SpatialDataCohort.to_anndata`).
       Only features whose ``var["feature_types"]`` is ``"Gene Expression"`` are kept, if the column exists. The
       shared var is the intersection of the members' Gene Expression Ensembl ids, sorted, so that every dataset has
       the same var. On a rerun the var of the datasets already written is binding, and a member that lacks any of
       its genes raises. What is left out is counted in the ``GeneKey`` record, the first of
       ``uns["sddb_contract"]["adapters"]`` (a step of this function, not an adapter: see
       :data:`~sddb.loader.MATERIALIZE_RECORDS`), and logged;
    2. the adapters run in order. An adapter with a ``fit`` method and no ``reference`` (:class:`StainNormalize`) is
       fit once on all members at that point of the chain, or takes the reference of the members already written;
    3. after the step that makes ``labels["cells"]`` (:class:`Rasterize`), the labels are written once to
       ``<dest>.sddb/scratch`` and reread from there, so that the later steps and the final write do not draw the
       cells again. They are deleted once the member is validated and marked done;
    4. the dataset is written with :func:`spatialbatch.write_collection_dataset` with ``uns["sddb_contract"]`` and
       ``uns["spatial"][uid]``, and checked with :func:`~sddb.contract.validate_dataset`. Then
       ``<dest>.sddb/done/<uid>`` is written, and a rerun skips the members that have it.

    Last, the whole collection is checked with :func:`~sddb.contract.validate`.

    Memory: the tables of all members not yet written are held in memory until each is written, since the shared var
    and a pooled fit need all of them. Images and labels stay lazy (dask) and are written block by block.

    Parameters
    ----------
    cohort
        The members: :class:`~sddb.Dataset` records, e.g. a :class:`~sddb.SpatialDataCohort`.
    dest
        The collection, a local zarr v3 group; created if missing.
    adapters
        The chain, e.g. :func:`~sddb.loader.default_chain`. It must hold one :class:`Rasterize` and, after it,
        :class:`~sddb.loader.Centroids`.
    role
        The image role of the collection, ``"he"`` or ``"if"``.
    roles
        Regexes for the image, the table and optionally existing labels of every member (see
        :meth:`LoaderItem.from_sdata`). Required until the catalog records each member's image roles.
    mpp_native
        Microns per scale-0 image pixel by uid. A member without an entry keeps its native grid.
    obs_columns
        Catalog columns copied into every dataset's ``obs``, as categoricals.
    runner
        ``"local"`` runs in this process; ``"snakemake"`` is not implemented yet.
    executor
        Only ``"local"`` with the local runner.
    workers
        Threads of the dask scheduler.
    allow_mixed_grids
        Allow a collection where some members are on the target grid and others (unknown mpp) on their native grid.

    Returns
    -------
    `dest` as a :class:`~pathlib.Path`.

    Raises
    ------
    ValueError
        On invalid arguments, a collection that would mix grids without `allow_mixed_grids`, a `dest` holding
        datasets of another cohort, or members that share no gene.
    ContractError
        If a written dataset or the collection does not conform to the contract.
    """
    if runner == "snakemake":
        raise NotImplementedError("runner='snakemake' is not implemented yet; use runner='local'")
    if runner != "local":
        raise ValueError(f"runner must be 'local' or 'snakemake', not {runner!r}")
    if executor != "local":
        raise ValueError(f"executor={executor!r} needs runner='snakemake'; the local runner runs in this process")
    if role not in ("he", "if"):
        raise ValueError(f"role must be 'he' or 'if', not {role!r}")
    if roles is None:
        raise ValueError("pass roles (the image and the table of each member): the catalog has no image-role field yet")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers}")
    adapters = list(adapters)
    rasterize = _rasterize_step(adapters)
    mpps = dict(mpp_native or {})

    members = sorted(cohort, key=lambda m: str(m.uid))
    uids = [str(m.uid) for m in members]
    if len(set(uids)) != len(uids):
        raise ValueError(f"the cohort repeats uids: {sorted({u for u in uids if uids.count(u) > 1})}")
    if missing := sorted({c for m in members for c in obs_columns if not hasattr(m, c)}):
        raise ValueError(f"obs_columns {missing} are not catalog columns")
    keys = {uid: f"dataset_{i}" for i, uid in enumerate(uids)}
    dest = Path(dest)
    root: Any = zarr.open_group(dest, mode="a", zarr_format=3, use_consolidated=False)
    if stale := sorted(k for k in root.group_keys() if _DATASET.match(k) and k not in keys.values()):
        raise ValueError(
            f"{dest} holds {stale}, which no member of this cohort maps to ({len(uids)} members); "
            "it was written for another cohort: use a new dest"
        )
    done = {uid for uid in uids if _is_done(dest, uid, keys[uid])}

    for uid in sorted(set(mpps) - set(uids)):
        logger.warning(f"mpp_native has {uid!r}, which is not a member of the cohort")
    grids = {
        uid: _contract(root[keys[uid]]).grid if uid in done else _grid(rasterize.mpp, mpps.get(uid)) for uid in uids
    }
    by_grid = {g: [u for u in uids if grids[u] == g] for g in ("target", "native")}
    if all(by_grid.values()) and not allow_mixed_grids:
        raise ValueError(
            f"the collection mixes grids: target for {by_grid['target']}, native (mpp unknown) for "
            f"{by_grid['native']}; pass mpp_native for those, or allow_mixed_grids=True"
        )
    for uid in by_grid["native"]:
        if uid not in done and rasterize.mpp is not None:
            logger.warning(f"{uid}: no mpp_native, so it keeps its native grid")

    todo = [m for m in members if str(m.uid) not in done]
    if todo:
        release = _release(cohort)
        written = [keys[u] for u in uids if u in done]
        with dask.config.set(num_workers=workers):
            _run(
                todo,
                dest,
                root,
                keys=keys,
                adapters=adapters,
                role=role,
                roles=roles,
                mpps=mpps,
                obs_columns=obs_columns,
                release=release,
                allow_mixed_grids=allow_mixed_grids,
                written=written,
            )
        shutil.rmtree(_state(dest) / "scratch", ignore_errors=True)
    assert_valid(dest)
    return dest


def _run(
    members: list[Any],
    dest: Path,
    root: Any,
    *,
    keys: Mapping[str, str],
    adapters: list[Any],
    role: Literal["he", "if"],
    roles: Mapping[str, str],
    mpps: Mapping[str, float],
    obs_columns: Sequence[str],
    release: str,
    allow_mixed_grids: bool,
    written: list[str],
) -> None:
    """Open, adapt and write the members that are not done.

    The steps up to the last one that needs a fit run on all members together, since a fit needs all of them; then
    each member runs the rest of the chain and is written before the next one, so that a rerun loses little.
    """
    items: dict[str, LoaderItem] = {}
    for m in members:
        uid = str(m.uid)
        items[uid] = LoaderItem.from_sdata(
            m.to_sdata(), uid=uid, study_id=str(m.study_id), role=role, roles=roles, mpp_native=mpps.get(uid)
        )
    items = _rekey_genes(items, root, written)

    fits = [i for i, step in enumerate(adapters) if _needs_fit(step)]
    split = fits[-1] if fits else 0
    scratch = _state(dest) / "scratch"
    for i, step in enumerate(adapters[:split]):
        if i in fits:
            _fit_or_reuse(step, list(items.values()), root, written)
        items = {uid: _build(step, item, scratch) for uid, item in items.items()}
    if fits:
        _fit_or_reuse(adapters[split], list(items.values()), root, written)

    by_uid = {str(m.uid): m for m in members}
    for uid in list(items):
        item = items.pop(uid)
        for step in adapters[split:]:
            item = _build(step, item, scratch)
        key = keys[uid]
        _write(root, dest, key, item, by_uid[uid], obs_columns, release, allow_mixed_grids)
        if issues := validate_dataset(root[key]):
            raise ContractError([f"{key} ({uid}): {issue}" for issue in issues])
        marker = _state(dest) / "done" / uid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"uid": uid, "key": key, "contract": CONTRACT_VERSION}))
        shutil.rmtree(scratch / uid, ignore_errors=True)


def _needs_fit(step: Any) -> bool:
    """A step with a ``fit`` method and no reference yet (:class:`StainNormalize`)."""
    return callable(getattr(step, "fit", None)) and getattr(step, "reference", "unset") is None


def _build(step: Any, item: LoaderItem, scratch: Path) -> LoaderItem:
    """Run one step; if it made new ``labels["cells"]``, write them to `scratch` once and reread them from there."""
    before = item.labels.get("cells")
    item = step.build(item)
    cells = item.labels.get("cells")
    if cells is None or cells is before:
        return item
    # one file per step, so that a step that derives new cells from persisted ones never overwrites what it reads
    path = scratch / item.uid / f"cells_{len(item.records)}.zarr"
    return dataclasses.replace(item, labels={**item.labels, "cells": _persist(cells, path)})


def _state(dest: Path) -> Path:
    """The run state of the collection at `dest`, beside it so that the collection holds only zarr groups."""
    return dest.parent / f"{dest.name}.sddb"


def _rasterize_step(adapters: Sequence[Any]) -> Rasterize:
    found = [a for a in adapters if isinstance(a, Rasterize)]
    if len(found) != 1:
        raise ValueError(f"the chain needs exactly one Rasterize step, it has {len(found)}")
    return found[0]


def _grid(mpp: float | None, mpp_native: float | None) -> Literal["target", "native"]:
    """The grid :class:`Rasterize` puts an item on."""
    return "target" if mpp is not None and mpp_native is not None else "native"


def _is_done(dest: Path, uid: str, key: str) -> bool:
    """Whether `uid` was written and validated at `key`; a marker for another key means the cohort changed."""
    marker = _state(dest) / "done" / uid
    if not marker.is_file():
        return False
    recorded = json.loads(marker.read_text()).get("key")
    if recorded != key:
        raise ValueError(
            f"{uid} was written as {recorded} in {dest}, but this cohort maps it to {key}: the cohort changed, use a "
            "new dest"
        )
    if not (dest / key).is_dir():
        raise ValueError(f"{dest} has a done marker for {uid} but no {key}; remove {marker} to rewrite it")
    return True


def _contract(group: Any) -> DatasetContract:
    uns: Any = read_elem(group["uns"]["sddb_contract"])
    return DatasetContract.from_uns(uns)


def _release(cohort: Any) -> str:
    """The catalog release the cohort was read from, ``"unknown"`` (logged) if it has none."""
    source = getattr(cohort, "_source", None)
    release = getattr(source, "release", None) or getattr(source, "version", None)
    if not release:
        logger.warning("the cohort has no catalog release; uns['sddb_contract']['catalog_release'] is 'unknown'")
        return "unknown"
    return str(release)


def _rekey_genes(items: dict[str, LoaderItem], root: Any, written: list[str]) -> dict[str, LoaderItem]:
    """Re-key every table to the shared Ensembl index and prepend the ``GeneKey`` record with what was left out.

    The index is the sorted intersection over the members, or, on a rerun, the var of the datasets already written,
    which every remaining member must then hold.
    """
    rekeyed: dict[str, tuple[AnnData, dict[str, int]]] = {}
    for uid, item in items.items():
        table = item.table
        n_other = 0
        if "feature_types" in table.var.columns:
            ge = (table.var["feature_types"].astype(str) == _GENE_EXPRESSION).to_numpy()
            n_other = int((~ge).sum())
            if n_other:
                logger.warning(f"{uid}: left out {n_other} features that are not {_GENE_EXPRESSION} (feature_types)")
            table = table[:, ge]
        table, n_drop, n_dup = _harmonize_var(table, "ensembl", uid)
        if n_drop or n_dup:
            logger.warning(f"{uid}: left out {n_drop} features without an Ensembl id and {n_dup} duplicate ids")
        rekeyed[uid] = table, {"non_gene_expression": n_other, "no_ensembl_id": n_drop, "duplicates": n_dup}

    if written:
        var = root[written[0]]["var"]
        shared = pd.Index(np.asarray(read_elem(var[var.attrs["_index"]])).astype(str))
        for uid, (table, _) in rekeyed.items():
            if lacking := shared.difference(table.var_names).tolist():
                raise ValueError(
                    f"{uid} lacks {len(lacking)} genes of the datasets already written (e.g. {lacking[:3]}), so the "
                    "collection's var cannot stay shared; use a new dest"
                )
    else:
        common = set.intersection(*(set(t.var_names) for t, _ in rekeyed.values()))
        shared = pd.Index(sorted(common))
    if not len(shared):
        raise ValueError("the members share no gene with an Ensembl id")

    out = {}
    for uid, (table, counts) in rekeyed.items():
        n_out = int(table.n_vars - len(shared))
        if n_out:
            logger.warning(f"{uid}: left out {n_out} genes that not every member has")
        params = {"gene_key": "ensembl", "feature_types": _GENE_EXPRESSION, **counts, "not_shared": n_out}
        record = AdapterRecord(name=_GENE_KEY, version=_GENE_KEY_VERSION, params=params)
        item = items[uid]
        out[uid] = dataclasses.replace(item, table=table[:, shared].copy(), records=[record, *item.records])
    return out


def _fit_or_reuse(step: Any, items: list[LoaderItem], root: Any, written: list[str]) -> None:
    """Fit `step` on `items`, or give it the reference recorded by the datasets already written."""
    if not written:
        step.fit(items)
        return
    records = [r for key in written for r in _contract(root[key]).adapters if r.name == step.name]
    if not records:
        raise ValueError(f"the datasets already written ({written}) did not run {step.name}; use a new dest")
    if any(r != records[0] for r in records):
        raise ValueError(f"the datasets already written ({written}) differ in their {step.name} record")
    rec = records[0]
    if isinstance(step, StainNormalize):
        rebuilt = StainNormalize.from_record(rec)
        if {k: v for k, v in rec.params.items() if k != "reference"} != step.config:
            raise ValueError(f"{step.name} {step.config} differs from the one the written datasets ran: {rec.params}")
        step.reference = rebuilt.reference
        logger.info(f"{step.name}: using the reference of the datasets already written")
        return
    raise ValueError(f"cannot take the fit of {step.name} from the written datasets; use a new dest")


def _persist(cells: xr.DataArray, path: Path) -> xr.DataArray:
    """Write ``labels["cells"]`` to `path` once, block by block, and return it reread lazily from there."""
    data = cells.data if isinstance(cells.data, da.Array) else da.from_array(np.asarray(cells.data))
    chunks = tuple(int(c) for c in data.chunksize)
    array = zarr.create_array(path, shape=data.shape, dtype=data.dtype, chunks=chunks, fill_value=0, overwrite=True)
    da.store(data.rechunk(chunks), array, lock=False)  # one dask block per zarr chunk: no two writes share one
    return xr.DataArray(da.from_array(array, chunks=chunks), dims=("y", "x"))


def _write(
    root: Any,
    dest: Path,
    key: str,
    item: LoaderItem,
    member: Any,
    obs_columns: Sequence[str],
    release: str,
    allow_mixed_grids: bool,
) -> None:
    """Write one item as ``root[key]``, replacing what an interrupted run left there."""
    missing = [
        what
        for what, ok in (
            ("labels['cells']", "cells" in item.labels),
            ("obsm['spatial'] (run Centroids)", "spatial" in item.table.obsm),
            ("a grid (run Rasterize)", item.grid is not None),
        )
        if not ok
    ]
    if missing:
        raise ValueError(f"{item.uid}: after the chain the item has no {', no '.join(missing)}")
    assert item.grid is not None
    contract = DatasetContract(
        version=CONTRACT_VERSION,
        uid=item.uid,
        study_id=item.study_id,
        catalog_release=release,
        role=item.role,
        grid=item.grid,
        mpp_target=item.mpp_target,
        mpp_native=item.mpp_native,
        upsampled=item.flags["upsampled"],
        allow_upsample=item.flags["allow_upsample"],
        allow_mixed_grids=allow_mixed_grids,
        dtype_rule=item.dtype_rule,
        channels=item.channels,
        adapters=item.records,
        stain_reference=item.stain_reference,
    )
    target = item.grid == "target"
    entry = SpatialEntry.model_validate(
        {
            "scalefactors": {
                "mpp": item.mpp_target if target else item.mpp_native,
                "mpp_native": item.mpp_native,
                "native_to_written": item.mpp_native / item.mpp_target
                if target and item.mpp_native and item.mpp_target
                else 1.0,
            },
            "metadata": {
                "role": item.role,
                "image": item.role,
                "labels": "cells",
                "tissue": "tissue" if "tissue" in item.labels else None,
            },
        }
    )

    table = item.table.copy()
    table.obs["uid"] = item.uid
    table.obs["study_id"] = item.study_id
    for col in obs_columns:
        table.obs[col] = pd.Categorical([getattr(member, col)] * table.n_obs)
    table.uns = {"sddb_contract": contract.to_uns(), "spatial": {item.uid: entry.model_dump()}}

    if (dest / key).exists():  # left by an interrupted run: it has no done marker
        shutil.rmtree(dest / key)
    write_collection_dataset(
        root,
        key=key,
        table=table,
        images={item.role: _dask(item.image)},
        labels={name: _dask(labels) for name, labels in item.labels.items()},
    )


def _dask(array: xr.DataArray) -> da.Array:
    return array.data if isinstance(array.data, da.Array) else da.from_array(np.asarray(array.data))
