"""What a rule of the workflow of ``materialize(runner="snakemake")`` runs.

``python -m sddb.loader._run_one fit STATE``, ``... member STATE UID`` and ``... validate STATE``, where ``STATE`` is
the run state beside the collection that holds ``config.yaml`` (see :mod:`sddb.loader._snakemake`). Each step calls
the functions of the in-process runner (:mod:`sddb.loader._materialize`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import dask
import yaml  # type: ignore[import-untyped]
import zarr

from sddb.contract import CONTRACT_VERSION, AdapterRecord
from sddb.loader._adapters import adapter_from_record
from sddb.loader._item import LoaderItem
from sddb.loader._materialize import _finalize, _fit_stage, _needs_fit, _read_store, _rekey, _write_member
from sddb.loader._snakemake import fit_key


def fit(config: Mapping[str, Any], state: Path) -> None:
    """Fit the chain's fit steps once on all members left to write, and write their records to ``stain.json``."""
    adapters = _chain(config)
    fits = [i for i, step in enumerate(adapters) if _needs_fit(step)]
    items = {uid: _open(config, uid) for uid in config["todo"]}
    root: Any = zarr.open_group(config["dest"], mode="r", use_consolidated=False)
    scratch = state / "fit"
    _fit_stage(adapters, items, root, config["written"], scratch)
    fitted = {str(i): adapters[i].to_record().model_dump() for i in fits}
    _write_json(state / "stain.json", {"inputs": fit_key(config), "records": fitted})
    shutil.rmtree(scratch, ignore_errors=True)


def member(config: Mapping[str, Any], state: Path, uid: str) -> None:
    """Run the whole chain on one member, with the fitted steps of ``stain.json``, and write it."""
    adapters = _chain(config)
    if config["fit"]:
        for i, record in json.loads((state / "stain.json").read_text())["records"].items():
            adapters[int(i)] = adapter_from_record(AdapterRecord(**record))
    _write_member(config, uid, _open(config, uid), adapters, state / "scratch")


def validate(config: Mapping[str, Any], state: Path) -> None:
    """Write the collection's root metadata, validate the collection and write ``valid.json``."""
    keys = _finalize(Path(config["dest"]))
    _write_json(state / "valid.json", {"datasets": keys, "contract": CONTRACT_VERSION})


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and run one step."""
    parser = argparse.ArgumentParser(prog="python -m sddb.loader._run_one", description=__doc__)
    parser.add_argument("step", choices=("fit", "member", "validate"))
    parser.add_argument("state", type=Path, help="the run state dir, which holds config.yaml")
    parser.add_argument("uid", nargs="?", help="the member, for the member step")
    parser.add_argument("--threads", type=int, default=1, help="threads of the dask scheduler")
    args = parser.parse_args(argv)
    if (args.step == "member") != (args.uid is not None):
        parser.error("a uid is given with the member step, and only with it")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = yaml.safe_load((args.state / "config.yaml").read_text())
    with dask.config.set(num_workers=args.threads):
        if args.step == "fit":
            fit(config, args.state)
        elif args.step == "member":
            member(config, args.state, args.uid)
        else:
            validate(config, args.state)


def _chain(config: Mapping[str, Any]) -> list[Any]:
    return [adapter_from_record(AdapterRecord(**record)) for record in config["adapters"]]


def _open(config: Mapping[str, Any], uid: str) -> LoaderItem:
    """Open one member from its store (``X`` lazily, see :func:`_read_store`), re-keyed to the collection's var."""
    m = config["members"][uid]
    item = LoaderItem.from_sdata(
        _read_store(m["path"]),
        uid=uid,
        study_id=m["study_id"],
        role=config["role"],
        roles=config["roles"],
        mpp_native=m["mpp_native"],
    )
    return _rekey(item, config["genes"], config["var"])


def _write_json(path: Path, content: Mapping[str, Any]) -> None:
    """Write `content` atomically: snakemake takes the file as the rule's output once it exists."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(dict(content)))
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
