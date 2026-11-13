"""Write a cohort as a loader collection (:mod:`sddb.contract`), in process: the ``runner="local"`` path."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import re
import shutil
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
import zarr
from anndata.io import read_elem
from zarr.errors import ZarrUserWarning

from sddb.concat import _ENSEMBL, _ENSEMBL_COLUMNS, _looks_ensembl
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
from sddb.loader._adapters import MATERIALIZE_RECORDS, Rasterize, StainNormalize
from sddb.loader._item import LoaderItem

if TYPE_CHECKING:
    from sddb.adapter import TaskAdapter

logger = logging.getLogger(__name__)

_DATASET = re.compile(r"^dataset_\d+$")
_GENE_EXPRESSION = "Gene Expression"
_GENE_KEY_VERSION = "0.3.0"
#: The columns of the collection's var besides ``present``, carried from the first member that has the feature.
_VAR_COLUMNS = ("feature_types", "gene_ids", "original_var_name")
_LOADER_EXTRA = "pip install 'spatialdata-db[loader]'"


def _spatialbatch() -> Any:
    """:mod:`spatialbatch`, imported only when a collection is written: it is not a dependency of ``import sddb``."""
    try:
        import spatialbatch  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            f"writing a loader collection needs spatialbatch, from the `loader` extra: {_LOADER_EXTRA}"
        ) from e
    return spatialbatch


def write_collection_dataset(*args: Any, **kwargs: Any) -> None:
    """:func:`spatialbatch.write_collection_dataset`, imported when called."""
    _spatialbatch().write_collection_dataset(*args, **kwargs)


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
    slurm: Mapping[str, Any] | None = None,
) -> Path:
    """Write every member of `cohort` as one dataset of a loader collection at `dest`, and validate it.

    Members are opened lazily (``to_sdata``) and written in uid order as ``dataset_0``, ``dataset_1``, ... The run
    state lives beside the collection, in ``<dest>.sddb/`` (``done/<uid>`` markers and ``scratch/``), so that `dest`
    holds only zarr groups. Per member:

    1. its table is put on the collection's var, the union of the members' features, sorted: the outer join of
       :meth:`~sddb.SpatialDataCohort.to_anndata` (``ad.concat(join="outer")`` in ``sddb/concat.py``). The
       ``"Gene Expression"`` features (``var["feature_types"]``; all of them without that column) are re-keyed to
       their Ensembl id (``gene_key="ensembl"``), kept in ``var["gene_ids"]``. A Gene Expression feature without an
       id keeps its name (``gene_ids`` null), as do the other features, e.g. control probes; repeated names, such as
       two rows with one id, are made unique (``A``, ``A-1``). Features join by that name. ``var["feature_types"]``,
       ``var["gene_ids"]`` and ``var["original_var_name"]`` (the source name, e.g. the gene symbol) are carried from
       the first member that has the feature. A source without ``var["feature_types"]`` has its features typed
       ``"Gene Expression"``, spatialdata-io's rule, and the GeneKey record says so (``feature_types_from_source``). A
       feature a member did not measure is 0 in its ``X``, and ``var["present"]`` of its dataset is False there; the
       contract counts both (``n_features_present``, ``n_features_filled``). On a rerun the var of the datasets
       already written is binding, and a member with a feature outside it raises. No feature is left out. The
       counts (features without an id, repeated ids, renamed, filled) go into the ``GeneKey`` record, the
       first of ``uns["sddb_contract"]["adapters"]`` (a step of this function, not an adapter: see
       :data:`~sddb.loader.MATERIALIZE_RECORDS`), and are logged;
    2. the adapters run in order. An adapter with a ``fit`` method and no ``reference`` (:class:`StainNormalize`) is
       fit once on all members at that point of the chain, or takes the reference of the members already written;
    3. after the step that makes ``labels["cells"]`` (:class:`Rasterize`), the labels are written once to
       ``<dest>.sddb/scratch`` and reread from there, so that the later steps and the final write do not draw the
       cells again. They are deleted once the member is validated and marked done;
    4. the dataset is written with :func:`spatialbatch.write_collection_dataset` with ``uns["sddb_contract"]`` and
       ``uns["spatial"][uid]`` into a store of its own in ``<dest>.sddb/scratch``, checked with
       :func:`~sddb.contract.validate_dataset` and moved into `dest`. Then ``<dest>.sddb/done/<uid>`` is written, and
       a rerun skips the members that have it.

    Last, the root of `dest` is marked as an annbatch collection, the metadata of all datasets is consolidated into
    it, and the whole collection is checked with :func:`~sddb.contract.validate`. Only this last step writes the root
    metadata, so members written in parallel never race on it.

    With ``runner="snakemake"`` the same steps run as a workflow, generated into ``<dest>.sddb/Snakefile`` and
    ``<dest>.sddb/config.yaml``: rule ``fit_stain`` (only if the chain fits a stain reference) writes
    ``<dest>.sddb/stain.json``, rule ``dataset_<uid>`` writes one member (its output is the done marker), and rule
    ``validate`` runs the last step and writes ``<dest>.sddb/valid.json``. The shared var and the dataset keys are
    fixed in ``config.yaml`` before any job runs, and each job rebuilds the chain from the adapters' records
    (:func:`~sddb.loader.adapter_from_record`), so every adapter must be a loader adapter.

    Memory: a member with a local store (a :class:`~sddb.Dataset`) is opened with its tables' ``X`` as dask arrays
    (:func:`_read_store`), as the snakemake jobs open it, so that the controller holds only obs and var of the members
    not yet written. ``X`` is computed once, by :func:`spatialbatch.write_collection_dataset` after it sorts the rows,
    so a member's write holds one ``X`` in memory. Images and labels stay lazy (dask) and are written block by block.

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
        ``"local"`` runs in this process; ``"snakemake"`` runs the generated workflow with snakemake.
    executor
        ``"local"``, or ``"slurm"`` with the snakemake runner (``snakemake --executor slurm``).
    workers
        Threads of the dask scheduler with the local runner; jobs at once with the snakemake runner (``--cores``
        locally, ``--jobs`` on Slurm), each running one dask thread.
    allow_mixed_grids
        Allow a collection where some members are on the target grid and others (unknown mpp) on their native grid.
    slurm
        Default resources of the Slurm jobs, passed as ``--default-resources``: ``runtime`` (minutes, or a span such
        as ``"2h"``) and exactly one of ``mem_mb`` and ``mem_mb_per_cpu`` (an ``int`` of MB) are required;
        ``slurm_partition``, ``slurm_account`` and the other resources of the snakemake Slurm executor are optional.
        ``slurm["fit_stain"]`` overrides them for the stain fit, which holds the tables of all members, e.g.
        ``{"mem_mb": 16000, "runtime": "2h", "fit_stain": {"mem_mb": 64000}}``. Only with ``executor="slurm"``. The
        jobs run on the compute nodes, so `dest`, its run state and every member's store (an absolute path) must be
        on a filesystem that the nodes share with this machine.

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
    RuntimeError
        If the snakemake workflow fails; the message names its log.
    """
    if runner not in ("local", "snakemake"):
        raise ValueError(f"runner must be 'local' or 'snakemake', not {runner!r}")
    if executor not in ("local", "slurm"):
        raise ValueError(f"executor must be 'local' or 'slurm', not {executor!r}")
    if runner == "local" and executor != "local":
        raise ValueError(f"executor={executor!r} needs runner='snakemake'; the local runner runs in this process")
    if slurm is not None and executor != "slurm":
        raise ValueError("slurm resources are for executor='slurm'")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers}")
    adapters = [copy.deepcopy(a) for a in adapters]  # a fit sets a reference: never on the caller's objects
    if runner == "snakemake":
        from sddb.loader import _snakemake

        _snakemake.records(adapters)  # checked before anything is opened: the jobs rebuild the chain from these
        if executor == "slurm":
            _snakemake.command(_state(Path(dest)), executor=executor, workers=workers, slurm=slurm)  # checks slurm
    plan, items = _prepare(
        cohort,
        dest,
        adapters=adapters,
        role=role,
        roles=roles,
        mpp_native=mpp_native,
        obs_columns=obs_columns,
        allow_mixed_grids=allow_mixed_grids,
    )
    if runner == "snakemake":
        del items  # each job opens its member again
        _snakemake.run(plan, adapters, executor=executor, workers=workers, slurm=slurm)
        return Path(dest)
    with dask.config.set(num_workers=workers):
        _run_local(plan, adapters, items)
    return Path(dest)


def _prepare(
    cohort: Iterable[Any],
    dest: str | Path,
    *,
    adapters: Sequence[Any],
    role: Literal["he", "if"],
    roles: Mapping[str, str] | None,
    mpp_native: Mapping[str, float] | None,
    obs_columns: Sequence[str],
    allow_mixed_grids: bool,
) -> tuple[dict[str, Any], dict[str, LoaderItem]]:
    """Check the cohort against `dest`, open the members not done and fix what every member must agree on.

    Both runners start here. The plan (plain JSON) holds the dataset key, the catalog values and the source of every
    member, the shared var, the members to write and whether the chain fits; the snakemake runner hands it to its
    jobs in ``config.yaml``, so that jobs running in parallel agree. The items are the members not done, opened and
    re-keyed to the shared var.
    """
    if role not in ("he", "if"):
        raise ValueError(f"role must be 'he' or 'if', not {role!r}")
    if roles is None:
        raise ValueError("pass roles (the image and the table of each member): the catalog has no image-role field yet")
    rasterize = _rasterize_step(adapters)
    mpps = dict(mpp_native or {})

    members = sorted(cohort, key=lambda m: str(m.uid))
    uids = [str(m.uid) for m in members]
    if len(set(uids)) != len(uids):
        raise ValueError(f"the cohort repeats uids: {sorted({u for u in uids if uids.count(u) > 1})}")
    if missing := sorted({c for m in members for c in obs_columns if not hasattr(m, c)}):
        raise ValueError(f"obs_columns {missing} are not catalog columns")
    keys = {uid: f"dataset_{i}" for i, uid in enumerate(uids)}
    dest = Path(dest).absolute()
    root: Any = zarr.open_group(dest, mode="a", zarr_format=3, use_consolidated=False)
    if stale := sorted(k for k in root.group_keys() if _DATASET.match(k) and k not in keys.values()):
        raise ValueError(
            f"{dest} holds {stale}, which no member of this cohort maps to ({len(uids)} members); "
            "it was written for another cohort: use a new dest"
        )
    done = {uid for uid in uids if _is_done(dest, uid, keys[uid])}

    written = [keys[u] for u in uids if u in done]
    _check_chain(adapters, root, written)

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
    items: dict[str, LoaderItem] = {}
    paths: dict[str, str | None] = {}
    for m in todo:
        uid = str(m.uid)
        store = m._local_store() if callable(getattr(m, "_local_store", None)) else None
        sdata = _read_store(store) if store is not None else m.to_sdata()
        paths[uid] = None if sdata.path is None else str(Path(sdata.path).absolute())
        items[uid] = LoaderItem.from_sdata(
            sdata, uid=uid, study_id=str(m.study_id), role=role, roles=roles, mpp_native=mpps.get(uid)
        )
    shared, columns = _shared_features(items, root, written) if items else ([], {c: [] for c in _VAR_COLUMNS})
    items = {uid: _rekey(item, shared, columns) for uid, item in items.items()}

    plan = {
        "dest": str(dest),
        "role": role,
        "roles": dict(roles),
        "release": _release(cohort) if todo else None,
        "allow_mixed_grids": allow_mixed_grids,
        "obs_columns": list(obs_columns),
        "genes": shared,
        "var": columns,
        "written": written,
        "todo": [str(m.uid) for m in todo],
        "fit": any(_needs_fit(a) for a in adapters),
        "members": {
            str(m.uid): {
                "key": keys[str(m.uid)],
                "study_id": str(m.study_id),
                "mpp_native": mpps.get(str(m.uid)),
                "obs": {c: _plain_value(getattr(m, c), c) for c in obs_columns},
                "path": paths.get(str(m.uid)),
            }
            for m in members
        },
    }
    return json.loads(json.dumps(plan)), items


def _plain_value(value: Any, column: str) -> Any:
    """A catalog value as plain JSON, so that both runners write the same ``obs``; missing values become ``None``."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str | bool | int | float):
        return value
    raise ValueError(f"obs_columns: {column} holds a {type(value).__name__}, not a str, number or bool")


def _run_local(plan: Mapping[str, Any], adapters: list[Any], items: dict[str, LoaderItem]) -> None:
    """The in-process runner: fit, then write the members one by one, then finalize the collection.

    The steps up to the last one that needs a fit run on all members together, since a fit needs all of them; then
    each member runs the rest of the chain and is written before the next one, so that a rerun loses little.
    """
    dest = Path(plan["dest"])
    scratch = _state(dest) / "scratch"
    if plan["todo"]:
        root: Any = zarr.open_group(dest, mode="r", use_consolidated=False)
        split, items = _fit_stage(adapters, items, root, plan["written"], scratch)
        for uid in plan["todo"]:
            _write_member(plan, uid, items.pop(uid), adapters[split:], scratch)
    _finalize(dest)


def _fit_stage(
    adapters: list[Any], items: dict[str, LoaderItem], root: Any, written: list[str], scratch: Path
) -> tuple[int, dict[str, LoaderItem]]:
    """Run the steps up to the last one that needs a fit on all `items`, fitting each such step on all of them.

    A fitted step is rebuilt from its record (:func:`_settle`), as the snakemake jobs rebuild it, so that both runners
    apply the same reference. Returns the index of the first step left to run per member, and the items there.
    """
    fits = [i for i, step in enumerate(adapters) if _needs_fit(step)]
    split = fits[-1] if fits else 0
    for i, step in enumerate(adapters[:split]):
        if i in fits:
            _fit_or_reuse(step, list(items.values()), root, written)
            _settle(step)
        items = {uid: _build(step, item, scratch) for uid, item in items.items()}
    if fits:
        _fit_or_reuse(adapters[split], list(items.values()), root, written)
        _settle(adapters[split])
    return split, items


def _settle(step: Any) -> None:
    """Give a fitted :class:`StainNormalize` the reference rebuilt from its record (float64 lists)."""
    if isinstance(step, StainNormalize):
        step.reference = StainNormalize.from_record(step.to_record()).reference


def _write_member(plan: Mapping[str, Any], uid: str, item: LoaderItem, steps: Sequence[Any], scratch: Path) -> None:
    """Run `steps` on one member and write it as its ``dataset_<i>``, validated, then mark it done.

    The dataset is written into its own store under `scratch` and moved into the collection once it validates, so
    that this never writes the collection's root metadata: members written in parallel do not race on it, and a
    dataset that does not conform never enters the collection. :func:`_finalize` writes the root metadata.
    """
    for step in steps:
        item = _build(step, item, scratch)
    dest = Path(plan["dest"])
    key = plan["members"][uid]["key"]
    stage = scratch / uid / "stage.zarr"
    if stage.exists():
        shutil.rmtree(stage)
    staged: Any = zarr.open_group(stage, mode="w", zarr_format=3)
    _write(staged, key, item, plan)
    if issues := validate_dataset(staged[key]):
        shutil.rmtree(stage)
        raise ContractError([f"{key} ({uid}): {issue}" for issue in issues])
    if (dest / key).exists():  # left by an interrupted run: it has no done marker
        shutil.rmtree(dest / key)
    shutil.move(stage / key, dest / key)  # a rename within one filesystem (scratch is beside dest)
    _mark_done(_state(dest) / "done" / uid, {"uid": uid, "key": key, "contract": CONTRACT_VERSION})
    shutil.rmtree(scratch / uid, ignore_errors=True)


def _finalize(dest: Path) -> list[str]:
    """Write the collection's root metadata once every member is written, and validate the collection.

    This marks the root as an annbatch collection (what :func:`spatialbatch.write_collection_dataset` sets on its
    group) and consolidates the metadata of every dataset into the root, then removes the scratch dir. Readers open
    the root with its consolidated metadata by default (``zarr.open_group(use_consolidated=None)``), so a dataset
    changed by other means needs :func:`zarr.consolidate_metadata` again.

    Returns
    -------
    The dataset keys of the collection.
    """
    from annbatch.io import V1_ENCODING  # type: ignore[import-untyped]

    root: Any = zarr.open_group(dest, mode="a", zarr_format=3, use_consolidated=False)
    root.update_attributes(V1_ENCODING)
    with warnings.catch_warnings():
        # consolidated metadata is what annbatch writes (annbatch/io.py write_sharded) and readers use
        warnings.filterwarnings("ignore", "Consolidated metadata is currently not part", ZarrUserWarning)
        zarr.consolidate_metadata(dest)
    shutil.rmtree(_state(dest) / "scratch", ignore_errors=True)
    assert_valid(dest)
    return sorted((k for k in root.group_keys() if _DATASET.match(k)), key=lambda k: int(k.split("_")[1]))


def _mark_done(marker: Path, content: Mapping[str, str]) -> None:
    """Write the done marker atomically: a temp file in the same dir, renamed over the marker."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name(f".{marker.name}.tmp")
    tmp.write_text(json.dumps(dict(content)))
    os.replace(tmp, marker)


def _record(step: Any) -> dict[str, Any]:
    """A step's record as a dataset stores it (params through JSON); a given stain reference is kept apart."""
    params = json.loads(json.dumps(dict(step.config)))
    return {"name": step.name, "version": step.version, "params": params, "reference": params.pop("reference", None)}


def _check_chain(adapters: Sequence[Any], root: Any, written: list[str]) -> None:
    """Refuse a rerun whose chain differs from the one the written datasets ran, before anything is written.

    Records are compared by name, version and params. The fitted stain reference is left out, since a rerun reuses
    it, unless this chain was given one.
    """
    now = [_record(a) for a in adapters]
    for key in written:
        before = []
        for rec in _contract(root[key]).adapters:
            if rec.name in MATERIALIZE_RECORDS:
                continue
            params = dict(rec.params)
            ref = params.pop("reference", None)
            before.append({"name": rec.name, "version": rec.version, "params": params, "reference": ref})
        if [r["name"] for r in before] != [r["name"] for r in now]:
            raise ValueError(
                f"{key} was written by the chain {[r['name'] for r in before]}, this run has "
                f"{[r['name'] for r in now]}; use a new dest"
            )
        for old, new in zip(before, now, strict=True):
            if old["version"] != new["version"]:
                raise ValueError(f"{key} ran {old['name']} {old['version']}, this run has {new['version']}")
            for param in sorted(old["params"].keys() | new["params"].keys()):
                was, is_ = old["params"].get(param, "<unset>"), new["params"].get(param, "<unset>")
                if was != is_:
                    raise ValueError(
                        f"{key} ran {old['name']} with {param}={was!r}, this run has {param}={is_!r}; use the same "
                        "chain or a new dest"
                    )
            if new["reference"] is not None and new["reference"] != old["reference"]:
                raise ValueError(f"{key} ran {old['name']} with another reference than the one given; use a new dest")


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
    try:
        recorded = json.loads(marker.read_text())["key"]
    except (OSError, ValueError, KeyError, TypeError) as e:  # json.JSONDecodeError is a ValueError
        logger.warning(f"{uid}: the done marker {marker} is unreadable ({e!r}), so {uid} is written again")
        return False
    if recorded != key:
        raise ValueError(
            f"{uid} was written as {recorded} in {dest}, but this cohort maps it to {key}: the cohort changed, use a "
            "new dest"
        )
    if not (dest / key).is_dir():
        raise ValueError(f"{dest} has a done marker for {uid} but no {key}; remove {marker} to rewrite it")
    return True


def _read_store(path: str | Path) -> Any:
    """A member's SpatialData store with each table's ``X`` as a dask array.

    ``sd.read_zarr`` reads a table whole (spatialdata 0.8.0 ``_io/io_table.py:22-23``, anndata's ``read_zarr``).
    The controller and the ``fit_stain`` job need only the tables' obs and var but open every member, so ``X`` stays
    lazy (``anndata.experimental.read_elem_lazy``) until :func:`spatialbatch.write_collection_dataset` computes it
    once, after it sorts the rows. The table keeps ``X``, ``obs``, ``var``, ``uns`` and
    ``obsm["spatial"]``, which is what :meth:`LoaderItem.from_sdata` keeps.
    """
    import anndata as ad
    import spatialdata as sd
    from anndata.experimental import read_elem_lazy
    from anndata.io import read_elem

    sdata = sd.read_zarr(path, selection=("images", "labels", "points", "shapes"))
    tables = Path(path) / "tables"
    for name in sorted(p.name for p in tables.iterdir() if p.is_dir()) if tables.is_dir() else []:
        g: Any = zarr.open_group(tables / name, mode="r")
        sdata.tables[name] = ad.AnnData(
            X=read_elem_lazy(g["X"]) if "X" in g else None,
            obs=read_elem(g["obs"]),
            var=read_elem(g["var"]),
            uns=read_elem(g["uns"]) if "uns" in g else None,
            obsm={"spatial": read_elem(g["obsm"]["spatial"])} if "obsm" in g and "spatial" in g["obsm"] else None,
        )
    return sdata


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


@dataclasses.dataclass(frozen=True)
class _Features:
    """A member's features on their way to the collection's var: every source column, under its new name."""

    names: pd.Index  # the new names, unique, in source order
    types: np.ndarray  # var["feature_types"] of each
    gene_ids: np.ndarray  # the Ensembl id of each Gene Expression feature, None where it has none
    original: np.ndarray  # the source var name of each, e.g. the gene symbol
    counts: dict[str, int]  # information: nothing is left out


def _ensembl_ids(var: pd.DataFrame, names: np.ndarray) -> np.ndarray:
    """Each row's Ensembl id (version suffix stripped), ``None`` where it has none.

    The source is the one ``to_anndata(gene_key="ensembl")`` takes (:func:`sddb.concat._harmonize_var`): the var
    names if they look like Ensembl ids, else the first such column of :data:`sddb.concat._ENSEMBL_COLUMNS`.
    """
    candidates = [pd.Series(names, dtype="string")]
    candidates += [pd.Series(var[c].to_numpy(), dtype="string") for c in _ENSEMBL_COLUMNS if c in var.columns]
    key = next((c for c in candidates if _looks_ensembl(c)), None)
    out = np.full(len(names), None, dtype=object)
    if key is None:
        return out
    key = key.str.strip().str.replace(r"\.\d+$", "", regex=True)
    usable = key.str.fullmatch(_ENSEMBL).fillna(False).astype(bool).to_numpy()
    out[usable] = key[usable].astype(str).to_numpy()
    return out


def _features(item: LoaderItem) -> _Features:
    """Every feature of the item: Gene Expression by its Ensembl id where it has one, the others by their name.

    A Gene Expression feature without an Ensembl id keeps its name. Repeated names, such as two rows with one id, are
    made unique the way anndata does (``anndata.utils.make_index_unique``, ``anndata/utils.py:226``: the second ``A``
    becomes ``A-1``, ``:265``). Nothing is left out; ``counts`` says how many rows had no id, repeated an id or were
    renamed.

    Raises
    ------
    ValueError
        If a feature keyed by its name has no name.
    """
    from anndata.utils import make_index_unique

    var = item.table.var
    n = item.table.n_vars
    typed = "feature_types" in var.columns
    if typed:
        types = var["feature_types"].astype(str).to_numpy(dtype=object)
    else:  # spatialdata-io's rule for an untyped feature (readers/_utils/_features.py:37-39, visium.py:257-258)
        types = np.full(n, _GENE_EXPRESSION, dtype=object)
    ge = types == _GENE_EXPRESSION
    names = np.asarray(item.table.var_names.astype(str), dtype=object)
    gene_ids = np.full(n, None, dtype=object)
    gene_ids[ge] = _ensembl_ids(var[ge], names[ge])
    has_id = pd.notna(gene_ids)
    if (blank := ~has_id & (pd.Series(names).str.strip() == "").to_numpy()).any():
        raise ValueError(f"{item.uid}: {int(blank.sum())} features without an Ensembl id have no var name")
    keys = np.where(has_id, gene_ids, names).astype(str)
    index = pd.Index(keys)
    counts = {
        "source": n,
        "no_ensembl_id": int((ge & ~has_id).sum()),
        "duplicates": int(pd.Series(gene_ids[has_id]).duplicated().sum()),
        "made_unique": int(index.duplicated().sum()),
        "feature_types_from_source": typed,
    }
    unique = make_index_unique(index)
    return _Features(names=unique, types=types, gene_ids=gene_ids, original=names, counts=counts)


def _shared_features(
    items: Mapping[str, LoaderItem], root: Any, written: list[str]
) -> tuple[list[str], dict[str, list[Any]]]:
    """The collection's var: the sorted union over the members, or the written var, and its :data:`_VAR_COLUMNS`.

    Features join by their name in the collection: the Ensembl id, else the source var name (``to_anndata`` joins
    by var name, ``ad.concat(join="outer")`` in ``sddb/concat.py``). A repeated name joins by its suffix, so the
    second row of an id in one member meets the second row of that id in another. Each column comes from the first
    member that has the feature.
    """
    if written:
        var = root[written[0]]["var"]
        if missing := [c for c in _VAR_COLUMNS if c not in var.attrs.get("column-order", [])]:
            raise ValueError(f"{written[0]} was written without var{missing} (an older materialize); use a new dest")
        names = [str(g) for g in np.asarray(read_elem(var[var.attrs["_index"]]))]
        columns = {c: [None if pd.isna(v) else str(v) for v in read_elem(var[c])] for c in _VAR_COLUMNS}
        return names, columns
    seen: dict[str, tuple[Any, ...]] = {}
    for uid, item in items.items():
        f = _features(item)
        for name, *values in zip(f.names, f.types, f.gene_ids, f.original, strict=True):
            if seen.setdefault(name, tuple(values))[0] != values[0]:
                logger.warning(
                    f"{uid}: {name} is {values[0]!r}, an earlier member has it as {seen[name][0]!r}, which is kept"
                )
    if not seen:
        raise ValueError("the members have no feature")
    names = sorted(seen)
    return names, {c: [seen[k][i] for k in names] for i, c in enumerate(_VAR_COLUMNS)}


def _rekey(item: LoaderItem, shared: Sequence[str], columns: Mapping[str, Sequence[Any]]) -> LoaderItem:
    """Put the item's table on the collection's var `shared`, 0 where it lacks a feature, and record the counts.

    Each source column is moved to its column of `shared` by one product with a 0/1 matrix, so ``X`` stays sparse,
    keeps its dtype, and stays lazy if it is a dask array.

    Raises
    ------
    ValueError
        If the item has a feature outside `shared`: on a rerun, outside the var of the datasets already written.
    """
    import anndata as ad

    uid, table = item.uid, item.table
    features = _features(item)
    counts = features.counts
    if counts["no_ensembl_id"] or counts["duplicates"]:
        logger.warning(
            f"{uid}: kept {counts['no_ensembl_id']} {_GENE_EXPRESSION} features without an Ensembl id (by name) and "
            f"{counts['duplicates']} repeats of an id (made unique)"
        )
    target = pd.Index(shared).get_indexer(features.names)
    if (outside := target < 0).any():
        raise ValueError(
            f"{uid} has {int(outside.sum())} features outside the var of the datasets already written (e.g. "
            f"{features.names[outside][:3].tolist()}), so the collection's var cannot stay shared; use a new dest"
        )
    present = np.zeros(len(shared), bool)
    present[target] = True
    n_filled = int((~present).sum())
    if n_filled:
        logger.warning(f"{uid}: {n_filled} of {len(shared)} features of the collection are not measured, set to 0")
    var = pd.DataFrame(
        {
            "feature_types": list(columns["feature_types"]),
            "gene_ids": pd.array(list(columns["gene_ids"]), dtype="string"),
            "original_var_name": pd.array(list(columns["original_var_name"]), dtype="string"),
            "present": present,
        },
        index=pd.Index(shared),
    )
    X = _move_columns(table.X, np.arange(table.n_vars), target, len(shared))
    params = {"gene_key": "ensembl", "join": "outer", **counts, "present": int(present.sum()), "filled": n_filled}
    record = AdapterRecord(name=_GENE_KEY, version=_GENE_KEY_VERSION, params=params)
    rekeyed = ad.AnnData(X=X, obs=table.obs, var=var, obsm=dict(table.obsm))
    return dataclasses.replace(item, table=rekeyed, records=[record, *item.records])


def _move_columns(X: Any, columns: np.ndarray, target: np.ndarray, n: int) -> Any:
    """``X`` with source column ``columns[i]`` at ``target[i]`` of `n` columns, and 0 in the others."""
    move = sp.csr_matrix(
        (np.ones(len(columns), dtype=X.dtype), (columns, target)), shape=(X.shape[1], n), dtype=X.dtype
    )
    if isinstance(X, da.Array):
        X = X.rechunk({1: -1})
        meta = X._meta if sp.issparse(X._meta) else np.empty((0, 0), dtype=X.dtype)
        return X.map_blocks(_times, move, chunks=(X.chunks[0], (n,)), dtype=X.dtype, meta=meta)
    return _times(X, move)


def _times(block: Any, move: sp.csr_matrix) -> Any:
    out = block @ move
    return out.tocsr() if sp.issparse(out) else np.asarray(out)


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


def _write(group: Any, key: str, item: LoaderItem, plan: Mapping[str, Any]) -> None:
    """Write one item as ``group[key]``, with its contract, ``uns["spatial"]`` and catalog values from `plan`."""
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
    present = item.table.var["present"].to_numpy(dtype=bool)
    contract = DatasetContract(
        version=CONTRACT_VERSION,
        uid=item.uid,
        study_id=item.study_id,
        catalog_release=plan["release"],
        role=item.role,
        grid=item.grid,
        mpp_target=item.mpp_target,
        mpp_native=item.mpp_native,
        upsampled=item.flags["upsampled"],
        allow_upsample=item.flags["allow_upsample"],
        allow_mixed_grids=plan["allow_mixed_grids"],
        dtype_rule=item.dtype_rule,
        channels=item.channels,
        adapters=item.records,
        stain_reference=item.stain_reference,
        n_centroid_fallback=item.n_centroid_fallback,
        n_outside_image=item.n_outside_image,
        n_features_present=int(present.sum()),
        n_features_filled=int((~present).sum()),
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

    import anndata as ad

    obs = item.table.obs.copy()
    obs["uid"] = item.uid
    obs["study_id"] = item.study_id
    for col, value in plan["members"][item.uid]["obs"].items():
        obs[col] = pd.Categorical([value] * len(obs))
    # X is shared, not copied: a dask X (from _read_store) is computed once by spatialbatch, after it sorts the rows
    table = ad.AnnData(
        X=item.table.X,
        obs=obs,
        var=item.table.var,
        obsm=dict(item.table.obsm),
        uns={"sddb_contract": contract.to_uns(), "spatial": {item.uid: entry.model_dump()}},
    )
    write_collection_dataset(
        group,
        key=key,
        table=table,
        images={item.role: _dask(item.image)},
        labels={name: _dask(labels) for name, labels in item.labels.items()},
    )


def _dask(array: xr.DataArray) -> da.Array:
    return array.data if isinstance(array.data, da.Array) else da.from_array(np.asarray(array.data))
