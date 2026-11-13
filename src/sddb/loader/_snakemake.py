"""The snakemake runner of :func:`~sddb.loader.materialize`: the workflow beside the collection, and its command.

What runs in each rule is :mod:`sddb.loader._run_one`, on the same functions as the in-process runner.
"""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

from sddb.contract import AdapterRecord
from sddb.loader._adapters import adapter_from_record
from sddb.loader._materialize import _state

#: Slurm resources every job needs: without them the executor submits without a memory or time limit.
SLURM_REQUIRED = ("mem_mb", "runtime")
_RESOURCE = re.compile(r"^[a-zA-Z_]\w*$")  # what snakemake accepts as a resource name (resources.py, Resources.parse)

_FIT = """

rule fit_stain:
    output:
        $stain,
    threads: 1
    params:
        python=PYTHON,
        state=STATE,
    shell:
        "{params.python:q} -m sddb.loader._run_one fit {params.state:q} --threads {threads}"
"""
_DATASET = """

rule $name:$input
    output:
        $marker,
    threads: 1
    params:
        python=PYTHON,
        state=STATE,
        uid=$uid,
    shell:
        "{params.python:q} -m sddb.loader._run_one member {params.state:q} {params.uid:q} --threads {threads}"
"""


def records(adapters: Sequence[Any]) -> list[dict[str, Any]]:
    """The chain as plain :class:`~sddb.contract.AdapterRecord` dicts, which the jobs rebuild it from.

    Raises
    ------
    ValueError
        If an adapter has no record or :func:`~sddb.loader.adapter_from_record` does not rebuild it to the same record.
    """
    out = []
    for adapter in adapters:
        name = getattr(adapter, "name", type(adapter).__name__)
        to_record = getattr(adapter, "to_record", None)
        if not callable(to_record):
            raise ValueError(
                f"{name}: the snakemake jobs rebuild the chain from records, and it has no to_record; use runner='local'"
            )
        record = _plain(to_record().model_dump())
        try:
            rebuilt = _plain(adapter_from_record(AdapterRecord(**record)).to_record().model_dump())
        except ValueError as e:
            raise ValueError(
                f"{name}: the snakemake jobs cannot rebuild it from its record ({e}); use runner='local'"
            ) from e
        if rebuilt != record:
            raise ValueError(f"{name}: rebuilt from its record it is {rebuilt}, not {record}; use runner='local'")
        out.append(record)
    return out


def write_workflow(plan: Mapping[str, Any], adapters: Sequence[Any]) -> Path:
    """Write ``Snakefile`` and ``config.yaml`` into the run state beside the collection, and return that dir.

    The config is the plan of :func:`~sddb.loader._materialize._prepare` plus the adapters' records. Outputs of an
    earlier run that the jobs would take as done are removed: ``stain.json``, ``valid.json`` and the done marker of
    every member left to write (an unreadable one, see ``_is_done``).
    """
    dest = Path(plan["dest"])
    state = _state(dest)
    if "{" in str(state) or "}" in str(state):
        raise ValueError(f"{state}: snakemake reads braces in rule outputs as wildcards; use a dest without them")
    if unbacked := [uid for uid in plan["todo"] if plan["members"][uid]["path"] is None]:
        raise ValueError(
            f"the snakemake jobs open each member from its zarr store, and to_sdata() of {unbacked} has none"
        )
    state.mkdir(parents=True, exist_ok=True)
    (state / "config.yaml").write_text(yaml.safe_dump({**plan, "adapters": records(adapters)}, sort_keys=False))
    (state / "Snakefile").write_text(_snakefile(plan, state))
    for stale in ("stain.json", "valid.json"):
        (state / stale).unlink(missing_ok=True)
    for uid in plan["todo"]:
        (state / "done" / uid).unlink(missing_ok=True)
    return state


def command(
    state: Path,
    *,
    executor: Literal["local", "slurm"],
    workers: int,
    slurm: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """The snakemake command line that runs the workflow in `state`.

    Locally ``--cores <workers>``; on Slurm ``--executor slurm --jobs <workers> --default-resources <slurm>``.
    Snakemake's own metadata goes into ``<state>/.snakemake`` (``--directory``).

    Raises
    ------
    ValueError
        If `slurm` is given without ``executor="slurm"``, or lacks :data:`SLURM_REQUIRED` with it.
    """
    cmd = [sys.executable, "-m", "snakemake", "--snakefile", str(state / "Snakefile")]
    cmd += ["--configfile", str(state / "config.yaml"), "--directory", str(state)]
    if dry_run:
        cmd.append("--dry-run")
    if executor == "local":
        if slurm is not None:
            raise ValueError("slurm resources are for executor='slurm'")
        return [*cmd, "--cores", str(workers)]
    if executor != "slurm":
        raise ValueError(f"executor must be 'local' or 'slurm', not {executor!r}")
    slurm = dict(slurm or {})
    if missing := [k for k in SLURM_REQUIRED if k not in slurm]:
        raise ValueError(
            f"executor='slurm' needs the slurm resources {' and '.join(SLURM_REQUIRED)}; missing {missing}"
        )
    return [*cmd, "--executor", "slurm", "--jobs", str(workers), "--default-resources", *_resources(slurm)]


def run(
    plan: Mapping[str, Any],
    adapters: Sequence[Any],
    *,
    executor: Literal["local", "slurm"],
    workers: int,
    slurm: Mapping[str, Any] | None,
) -> None:
    """Write the workflow and run it with snakemake; its output goes to ``<state>/snakemake.log``.

    Raises
    ------
    RuntimeError
        If snakemake fails, with the end of its log.
    """
    state = write_workflow(plan, adapters)
    cmd = command(state, executor=executor, workers=workers, slurm=slurm)
    log = state / "snakemake.log"
    with log.open("w") as f:
        done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if done.returncode:
        tail = "\n".join(log.read_text().splitlines()[-30:])
        raise RuntimeError(f"snakemake failed (exit {done.returncode}); its log is {log}:\n{tail}")


def _resources(slurm: Mapping[str, Any]) -> list[str]:
    """``NAME=VALUE`` for ``--default-resources``: numbers as they are, strings as Python literals.

    Snakemake evaluates a value that is not an integer as a Python expression (``Resource.cli_evaluator``), so a
    string is quoted; it strips the quotes of ``mem``/``runtime`` spans such as ``'2h'`` before parsing them
    (``Resource.parse_human_friendly``).
    """
    out = []
    for name, value in slurm.items():
        if not _RESOURCE.match(name):
            raise ValueError(f"slurm resource {name!r} is not a valid name")
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(f"slurm resource {name} must be a number or a string, not {value!r}")
        out.append(f"{name}={value!r}" if isinstance(value, str) else f"{name}={value}")
    return out


def _snakefile(plan: Mapping[str, Any], state: Path) -> str:
    fit = bool(plan["fit"] and plan["todo"])
    stain = repr(str(state / "stain.json"))
    rules = string.Template(_FIT).substitute(stain=stain) if fit else ""
    names: dict[str, str] = {}
    for uid in plan["todo"]:
        name = "dataset_" + re.sub(r"\W", "_", uid)
        if name in names:
            raise ValueError(f"uids {names[name]!r} and {uid!r} give the same rule name {name}")
        names[name] = uid
        rules += string.Template(_DATASET).substitute(
            name=name,
            input=f"\n    input:\n        {stain}," if fit else "",
            marker=repr(str(state / "done" / uid)),
            uid=repr(uid),
        )
    template = resources.files("sddb.loader").joinpath("Snakefile.tmpl").read_text()
    done = "\n".join(f"        {str(state / 'done' / uid)!r}," for uid in plan["members"])
    return string.Template(template).substitute(
        dest=plan["dest"],
        python=repr(sys.executable),
        state=repr(str(state)),
        done=done,
        valid=repr(str(state / "valid.json")),
        rules=rules,
    )


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))
