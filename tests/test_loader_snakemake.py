"""``materialize(runner="snakemake")`` (Task 5 of T144): the generated workflow, run locally, against the in-process run."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml
import zarr
from tests.test_loader_materialize import ROLES, FakeCohort, FakeMember, _contract, write_member

from sddb.contract import AdapterRecord, validate
from sddb.loader import (
    Centroids,
    Rasterize,
    StainNormalize,
    TissueMask,
    _materialize,
    _snakemake,
    adapter_from_record,
    default_chain,
    materialize,
)

MPP = {"uidA": 0.25, "uidB": 0.25, "uidC": 0.25}
REFERENCE = {
    "method": "macenko",
    "stain_matrix": [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
    "white_point": [255.0, 255.0, 255.0],
    "max_concentrations": [1.9, 1.0],
}


@pytest.fixture
def cohort(tmp_path):
    a = write_member(tmp_path, "uidA", seed=0)
    b = write_member(tmp_path, "uidB", seed=1, extra_gene=True)
    return FakeCohort([FakeMember("uidB", b[0]), FakeMember("uidA", a[0])])


def _job_stats(log: str) -> dict[str, int]:
    """The ``Job stats:`` table snakemake prints (``DAG.stats``), as {job: count}."""
    lines = log[log.index("Job stats:") :].splitlines()[3:]  # the title, the header and the dashes
    stats = {}
    for line in lines:
        if not line.strip():
            break
        job, count = line.split()
        stats[job] = int(count)
    return stats


def _nodes(group, prefix=""):
    """Every node below `group` by path: arrays as (numpy, attrs), groups as attrs. ``uns`` is among them."""
    out = {}
    for name, node in group.members():
        path = f"{prefix}{name}"
        if isinstance(node, zarr.Array):
            out[path] = (np.asarray(node[...]), dict(node.attrs))
        else:
            out[path] = dict(node.attrs)
            out.update(_nodes(node, f"{path}/"))
    return out


def test_snakefile_dry_run_lists_rules(cohort, tmp_path):
    dest = tmp_path / "c.zarr"
    chain = default_chain(0.5)
    plan, _ = _materialize._prepare(
        cohort, dest, adapters=chain, role="he", roles=ROLES, mpp_native=MPP, obs_columns=(), allow_mixed_grids=False
    )
    state = _snakemake.write_workflow(plan, chain)
    assert state == tmp_path / "c.zarr.sddb"
    assert (state / "Snakefile").is_file()
    assert (state / "config.yaml").is_file()

    cmd = _snakemake.command(state, executor="local", workers=1, dry_run=True)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stats = _job_stats(out.stdout + out.stderr)
    assert stats == {"validate": 1, "dataset_uidA": 1, "dataset_uidB": 1, "fit_stain": 1, "total": 4}
    assert not any(dest.glob("dataset_*"))

    # a given stain reference needs no fit rule
    given = [*chain[:3], StainNormalize(reference=StainNormalize.from_record(_stain_record()).reference)]
    plan, _ = _materialize._prepare(
        cohort, dest, adapters=given, role="he", roles=ROLES, mpp_native=MPP, obs_columns=(), allow_mixed_grids=False
    )
    state = _snakemake.write_workflow(plan, given)
    out = subprocess.run(_snakemake.command(state, executor="local", workers=1, dry_run=True), **_capture())
    assert "fit_stain" not in _job_stats(out.stdout + out.stderr)


def _capture():
    return {"capture_output": True, "text": True, "check": True}


def _stain_record():
    return AdapterRecord(
        name="StainNormalize",
        version="0.1.0",
        params={"method": "macenko", "fit_pixels": 10_000, "reference": REFERENCE},
    )


def test_commands(tmp_path):
    state = tmp_path / "c.zarr.sddb"
    head = [sys.executable, "-m", "snakemake", "--snakefile", str(state / "Snakefile")]
    head += ["--configfile", str(state / "config.yaml"), "--directory", str(state)]
    assert _snakemake.command(state, executor="local", workers=3) == [*head, "--cores", "3"]
    slurm = {"mem_mb": 16000, "runtime": "2h", "slurm_partition": "cpu_p"}
    assert _snakemake.command(state, executor="slurm", workers=8, slurm=slurm) == [
        *head,
        "--executor",
        "slurm",
        "--jobs",
        "8",
        "--default-resources",
        "mem_mb=16000",
        "runtime='2h'",
        "slurm_partition='cpu_p'",
    ]
    with pytest.raises(ValueError, match="mem_mb.*runtime"):
        _snakemake.command(state, executor="slurm", workers=8, slurm={"slurm_partition": "cpu_p"})
    with pytest.raises(ValueError, match="slurm"):
        _snakemake.command(state, executor="local", workers=1, slurm=slurm)


def test_adapter_record_roundtrip():
    chain = [*default_chain(0.5, allow_upsample=True), StainNormalize.from_record(_stain_record())]
    records = yaml.safe_load(yaml.safe_dump(_snakemake.records(chain)))
    rebuilt = [adapter_from_record(AdapterRecord(**r)) for r in records]
    assert [type(a) for a in rebuilt] == [type(a) for a in chain]
    assert [a.to_record() for a in rebuilt] == [a.to_record() for a in chain]
    assert rebuilt[-1].reference is not None
    np.testing.assert_array_equal(rebuilt[-1].reference.stain_matrix, chain[-1].reference.stain_matrix)

    class Custom:
        name, version, config = "Custom", "1.0.0", {}

        def build(self, item):
            return item

    with pytest.raises(ValueError, match="Custom.*rebuild"):
        _snakemake.records([Rasterize(0.5), Custom()])


def test_snakemake_local_equals_inprocess(cohort, tmp_path):
    kwargs = {"adapters": default_chain(0.5), "role": "he", "roles": ROLES, "mpp_native": MPP}
    local = materialize(cohort, tmp_path / "local.zarr", obs_columns=["tissue"], **kwargs)
    snake = materialize(cohort, tmp_path / "snake.zarr", obs_columns=["tissue"], runner="snakemake", **kwargs)
    assert validate(snake) == []
    state = tmp_path / "snake.zarr.sddb"
    assert {"Snakefile", "config.yaml", "stain.json", "valid.json"} <= {p.name for p in state.iterdir()}
    assert sorted(p.name for p in snake.iterdir()) == ["dataset_0", "dataset_1", "zarr.json"]
    assert json.loads((state / "valid.json").read_text())["datasets"] == ["dataset_0", "dataset_1"]

    for key in ("dataset_0", "dataset_1"):
        a, b = zarr.open_group(local / key, mode="r"), zarr.open_group(snake / key, mode="r")
        nodes_a, nodes_b = _nodes(a), _nodes(b)
        assert sorted(nodes_a) == sorted(nodes_b)
        assert {"images/he", "labels/cells", "labels/tissue", "uns/sddb_contract"} <= set(nodes_a)
        for path, node in nodes_a.items():
            if isinstance(node, dict):
                assert node == nodes_b[path], path
                continue
            (array, attrs), (other, other_attrs) = node, nodes_b[path]
            assert array.dtype == other.dtype, path
            np.testing.assert_array_equal(array, other, err_msg=path)
            assert attrs == other_attrs, path
        assert dict(a.attrs) == dict(b.attrs)
        assert _contract(local, key) == _contract(snake, key)


def test_parallel_members_consolidated(tmp_path):
    members = FakeCohort(
        FakeMember(uid, write_member(tmp_path, uid, seed=i)[0]) for i, uid in enumerate(("uidA", "uidB", "uidC"))
    )
    dest = tmp_path / "c.zarr"
    chain = [Rasterize(0.5), Centroids(), TissueMask()]
    materialize(members, dest, adapters=chain, role="he", roles=ROLES, mpp_native=MPP, runner="snakemake", workers=2)
    assert validate(dest) == []

    meta = json.loads((dest / "zarr.json").read_text())
    listed = meta["consolidated_metadata"]["metadata"]
    for key in ("dataset_0", "dataset_1", "dataset_2"):
        assert {key, f"{key}/X", f"{key}/images/he", f"{key}/labels/cells", f"{key}/labels/tissue"} <= set(listed)
        assert json.loads((dest / key / "zarr.json").read_text()).get("consolidated_metadata") is None
    assert meta["attributes"]["encoding-type"] == "annbatch-preshuffled"
    root = zarr.open_group(dest, mode="r", use_consolidated=True)
    assert sorted(root.group_keys()) == ["dataset_0", "dataset_1", "dataset_2"]
    assert sorted(p.name for p in (tmp_path / "c.zarr.sddb" / "done").iterdir()) == ["uidA", "uidB", "uidC"]
    # two members were written at once: two dataset jobs started before the first one finished
    log = (tmp_path / "c.zarr.sddb" / "snakemake.log").read_text()
    assert log[: log.index("Finished jobid")].count("rule dataset_") == 2
