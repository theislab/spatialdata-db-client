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
import xarray as xr
import zarr
from annbatch.io import V1_ENCODING  # type: ignore[import-untyped]
from anndata.io import read_elem
from spatialbatch import write_collection_dataset  # type: ignore[import-untyped]
from zarr.errors import ZarrUserWarning

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
from sddb.loader._adapters import MATERIALIZE_RECORDS, Rasterize, StainNormalize
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
    slurm: Mapping[str, Any] | None = None,
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

    Memory: the tables of all members not yet written are held in memory until each is written, since the shared var
    and a pooled fit need all of them; a member whose ``to_sdata`` gives a dask ``X`` keeps it lazy until its write.
    The snakemake jobs read ``X`` so (``_run_one._read_store``), and each member job holds one table in memory.
    Images and labels stay lazy (dask) and are written block by block.

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
        sdata = m.to_sdata()
        paths[uid] = None if sdata.path is None else str(Path(sdata.path).absolute())
        items[uid] = LoaderItem.from_sdata(
            sdata, uid=uid, study_id=str(m.study_id), role=role, roles=roles, mpp_native=mpps.get(uid)
        )
    shared = _shared_genes(items, root, written) if items else []
    items = {uid: _rekey(item, shared) for uid, item in items.items()}

    plan = {
        "dest": str(dest),
        "role": role,
        "roles": dict(roles),
        "release": _release(cohort) if todo else None,
        "allow_mixed_grids": allow_mixed_grids,
        "obs_columns": list(obs_columns),
        "genes": shared,
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


def _gene_table(item: LoaderItem) -> tuple[AnnData, dict[str, int]]:
    """The item's Gene Expression features keyed by Ensembl id, and the counts of what was left out."""
    table = item.table
    n_other = 0
    if "feature_types" in table.var.columns:
        ge = (table.var["feature_types"].astype(str) == _GENE_EXPRESSION).to_numpy()
        n_other = int((~ge).sum())
        table = table[:, ge]
    table, n_drop, n_dup = _harmonize_var(table, "ensembl", item.uid)
    return table, {"non_gene_expression": n_other, "no_ensembl_id": n_drop, "duplicates": n_dup}


def _shared_genes(items: Mapping[str, LoaderItem], root: Any, written: list[str]) -> list[str]:
    """The collection's var: the sorted intersection over the members, or the var of the datasets already written."""
    if written:
        var = root[written[0]]["var"]
        shared = [str(g) for g in np.asarray(read_elem(var[var.attrs["_index"]]))]
    else:
        shared = sorted(set.intersection(*(set(_gene_table(item)[0].var_names) for item in items.values())))
    if not shared:
        raise ValueError("the members share no gene with an Ensembl id")
    return shared


def _rekey(item: LoaderItem, shared: Sequence[str]) -> LoaderItem:
    """Re-key the item's table to the shared Ensembl index and prepend the ``GeneKey`` record with what was left out.

    Raises
    ------
    ValueError
        If the item lacks a gene of `shared`: on a rerun, a gene of the datasets already written.
    """
    uid = item.uid
    table, counts = _gene_table(item)
    if counts["non_gene_expression"]:
        logger.warning(
            f"{uid}: left out {counts['non_gene_expression']} features that are not {_GENE_EXPRESSION} (feature_types)"
        )
    if counts["no_ensembl_id"] or counts["duplicates"]:
        logger.warning(
            f"{uid}: left out {counts['no_ensembl_id']} features without an Ensembl id and {counts['duplicates']} "
            "duplicate ids"
        )
    if lacking := pd.Index(shared).difference(table.var_names).tolist():
        raise ValueError(
            f"{uid} lacks {len(lacking)} genes of the datasets already written (e.g. {lacking[:3]}), so the "
            "collection's var cannot stay shared; use a new dest"
        )
    n_out = int(table.n_vars - len(shared))
    if n_out:
        logger.warning(f"{uid}: left out {n_out} genes that not every member has")
    params = {"gene_key": "ensembl", "feature_types": _GENE_EXPRESSION, **counts, "not_shared": n_out}
    record = AdapterRecord(name=_GENE_KEY, version=_GENE_KEY_VERSION, params=params)
    return dataclasses.replace(item, table=table[:, list(shared)].copy(), records=[record, *item.records])


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
    if isinstance(table.X, da.Array):  # a lazy X (the snakemake jobs read it so): annbatch writes it from memory
        table.X = table.X.compute()
    table.obs["uid"] = item.uid
    table.obs["study_id"] = item.study_id
    for col, value in plan["members"][item.uid]["obs"].items():
        table.obs[col] = pd.Categorical([value] * table.n_obs)
    table.uns = {"sddb_contract": contract.to_uns(), "spatial": {item.uid: entry.model_dump()}}
    write_collection_dataset(
        group,
        key=key,
        table=table,
        images={item.role: _dask(item.image)},
        labels={name: _dask(labels) for name, labels in item.labels.items()},
    )


def _dask(array: xr.DataArray) -> da.Array:
    return array.data if isinstance(array.data, da.Array) else da.from_array(np.asarray(array.data))
