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

#: The memory resources of the Slurm executor (``--mem`` and ``--mem-per-cpu``): a job needs exactly one.
SLURM_MEMORY = ("mem_mb", "mem_mb_per_cpu")
#: The key of `slurm` whose resources override the defaults for the ``fit_stain`` job, which holds all tables.
SLURM_FIT = "fit_stain"
_RESOURCE = re.compile(r"^[a-zA-Z_]\w*$")  # what snakemake accepts as a resource name (resources.py, Resources.parse)
# what snakemake reads in a path: braces as wildcards, and it warns of surrounding whitespace, line breaks and
# double slashes (snakemake/io/__init__.py, check); a slash would also nest the uid's marker and scratch dirs
_UID_UNSAFE = re.compile(r"[{}/\\\n\r\t\0]|^\s|\s$")

_FIT = """

rule fit_stain:
    output:
        $stain,
    log:
        $log,
    threads: 1$resources
    params:
        python=PYTHON,
        state=STATE,
    shell:
        "{params.python:q} -m sddb.loader._run_one fit {params.state:q} --threads {threads} > {log:q} 2>&1"
"""
_DATASET = """

rule $name:$input
    output:
        $marker,
    log:
        $log,
    threads: 1
    params:
        python=PYTHON,
        state=STATE,
        uid=$uid,
    shell:
        "{params.python:q} -m sddb.loader._run_one member {params.state:q} {params.uid:q} --threads {threads} "
        "> {log:q} 2>&1"
"""


def fit_key(config: Mapping[str, Any]) -> dict[str, Any]:
    """What a stain fit depends on: the chain's records and the members' uids.

    Not their stores: a written member's store is ``None`` in the plan of the next run.
    """
    return {"adapters": config["adapters"], "members": sorted(config["members"])}


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


def write_workflow(
    plan: Mapping[str, Any],
    adapters: Sequence[Any],
    *,
    executor: Literal["local", "slurm"] = "local",
    slurm: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``Snakefile`` and ``config.yaml`` into the run state beside the collection, and return that dir.

    The config is the plan of :func:`~sddb.loader._materialize._prepare` plus the adapters' records, with every
    member's store as an absolute path. Outputs of an earlier run that the jobs would take as done are removed:
    ``valid.json``, the done marker of every member left to write (an unreadable one, see ``_is_done``), and
    ``stain.json`` unless it was fitted on the same chain and members (:func:`fit_key`), so a rerun after a failed
    member does not fit the stain reference again. The ``fit_stain`` resources of `slurm` become that rule's ``resources``.

    Raises
    ------
    ValueError
        If a path or uid holds what snakemake reads specially, a member to write has no store or its store does not
        exist, or, with ``executor="slurm"``, a store path is relative.
    """
    dest = Path(plan["dest"])
    state = _state(dest)
    if "{" in str(state) or "}" in str(state):
        raise ValueError(f"{state}: snakemake reads braces in rule outputs as wildcards; use a dest without them")
    if unsafe := [uid for uid in plan["members"] if _UID_UNSAFE.search(uid)]:
        raise ValueError(
            f"uids {unsafe} hold a brace, slash, line break or surrounding whitespace, which snakemake reads specially "
            "in the done marker's path; use runner='local'"
        )
    if unbacked := [uid for uid in plan["todo"] if plan["members"][uid]["path"] is None]:
        raise ValueError(
            f"the snakemake jobs open each member from its zarr store, and to_sdata() of {unbacked} has none"
        )
    members = {uid: dict(m) for uid, m in plan["members"].items()}
    for uid in plan["todo"]:
        path = Path(members[uid]["path"])
        if executor == "slurm" and not path.is_absolute():
            raise ValueError(f"{uid}: with executor='slurm' the store must be an absolute path, not {path}")
        if not path.exists():
            raise ValueError(f"{uid}: its store {path} does not exist")
        members[uid]["path"] = str(path.absolute())  # the jobs run in the state dir (--directory)
    fit_resources = _check_slurm(slurm)[1] if executor == "slurm" else {}
    state.mkdir(parents=True, exist_ok=True)
    config = {**plan, "members": members, "adapters": records(adapters)}
    (state / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (state / "Snakefile").write_text(_snakefile(plan, state, fit_resources))
    (state / "valid.json").unlink(missing_ok=True)
    stain = state / "stain.json"
    if stain.exists() and json.loads(stain.read_text()).get("inputs") != fit_key(config):
        stain.unlink()
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

    Locally ``--cores <workers>``; on Slurm ``--executor slurm --jobs <workers> --default-resources <slurm>``,
    without the ``fit_stain`` overrides, which the Snakefile sets on that rule. Snakemake's own metadata goes into
    ``<state>/.snakemake`` (``--directory``).

    Raises
    ------
    ValueError
        If `slurm` is given without ``executor="slurm"``, or is not valid with it (see :func:`_check_slurm`).
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
    defaults, _ = _check_slurm(slurm)
    return [*cmd, "--executor", "slurm", "--jobs", str(workers), "--default-resources", *_resources(defaults)]


def run(
    plan: Mapping[str, Any],
    adapters: Sequence[Any],
    *,
    executor: Literal["local", "slurm"],
    workers: int,
    slurm: Mapping[str, Any] | None,
) -> None:
    """Write the workflow and run it with snakemake.

    Snakemake's output goes to ``<state>/snakemake.log``, each rule's to ``<state>/logs/<rule>.log``.

    Raises
    ------
    RuntimeError
        If snakemake fails. It names the log of each failed rule and holds its last 30 lines, or the last 30 lines of
        snakemake's own log if no rule failed.
    """
    state = write_workflow(plan, adapters, executor=executor, slurm=slurm)
    cmd = command(state, executor=executor, workers=workers, slurm=slurm)
    log = state / "snakemake.log"
    with log.open("w") as f:
        done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if not done.returncode:
        return
    text = log.read_text()
    # snakemake's error line for a failed job (snakemake/logging.py, "Error in rule {rule_name}:")
    failed = list(dict.fromkeys(re.findall(r"Error in rule (\w+):", text)))
    parts = [f"snakemake failed (exit {done.returncode}); its log is {log}"]
    for rule in failed:
        rule_log = _log(state, rule)
        tail = _tail(rule_log.read_text()) if rule_log.is_file() else "(no log written)"
        parts.append(f"rule {rule} failed, its log is {rule_log}:\n{tail}")
    if not failed:
        parts.append(_tail(text))
    raise RuntimeError("\n".join(parts))


def _tail(text: str, n: int = 30) -> str:
    return "\n".join(text.splitlines()[-n:])


def _log(state: Path, rule: str) -> Path:
    """The log of a rule: ``<state>/logs/<rule>.log``; the rule of a member is named ``dataset_<uid>``."""
    return state / "logs" / f"{rule}.log"


def _check_slurm(slurm: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split `slurm` into the default resources and the ``fit_stain`` overrides, and check both.

    The defaults need ``runtime`` and exactly one of :data:`SLURM_MEMORY`. Memory is an ``int`` of MB: snakemake
    parses size spans only for ``mem`` and ``disk`` (resources.py, ``SizedResources``), not for ``mem_mb``.
    """
    defaults = dict(slurm or {})
    fit = defaults.pop(SLURM_FIT, None) or {}
    if not isinstance(fit, Mapping):
        raise ValueError(f"slurm[{SLURM_FIT!r}] must be a mapping of resources, not {fit!r}")
    fit = dict(fit)
    if SLURM_FIT in fit:
        raise ValueError(f"slurm[{SLURM_FIT!r}] cannot hold {SLURM_FIT!r}")
    memory = [k for k in SLURM_MEMORY if k in defaults]
    if len(memory) != 1 or "runtime" not in defaults:
        raise ValueError(
            "executor='slurm' needs the slurm resources runtime and exactly one of mem_mb and mem_mb_per_cpu; "
            f"got {sorted(defaults)}"
        )
    for where, resources_ in (("slurm", defaults), (f"slurm[{SLURM_FIT!r}]", fit)):
        for name in SLURM_MEMORY:
            value = resources_.get(name)
            if name in resources_ and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{where}: {name} must be a positive int of MB, not {value!r}")
        _resources(resources_)  # checks the names and value types
    return defaults, fit


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


def _snakefile(plan: Mapping[str, Any], state: Path, fit_resources: Mapping[str, Any]) -> str:
    fit = bool(plan["fit"] and plan["todo"])
    stain = repr(str(state / "stain.json"))
    lines = "".join(f"\n        {name}={value!r}," for name, value in fit_resources.items())
    rules = ""
    if fit:
        rules = string.Template(_FIT).substitute(
            stain=stain,
            log=repr(str(_log(state, "fit_stain"))),
            resources=f"\n    resources:{lines}" if lines else "",
        )
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
            log=repr(str(_log(state, name))),
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
        valid_log=repr(str(_log(state, "validate"))),
        rules=rules,
    )


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))
