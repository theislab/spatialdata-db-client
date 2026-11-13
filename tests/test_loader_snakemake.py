"""``materialize(runner="snakemake")``: the generated workflow, run locally, against the in-process run."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

# the loader extra
pytest.importorskip("pydantic")
pytest.importorskip("squidpy")
pytest.importorskip("dask_image")
pytest.importorskip("spatialbatch")
pytest.importorskip("snakemake")
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
    # the fit_stain overrides go to that rule, not to the defaults; mem_mb_per_cpu instead of mem_mb
    per_cpu = {"mem_mb_per_cpu": 4000, "runtime": 60, "fit_stain": {"mem_mb_per_cpu": 16000}}
    assert _snakemake.command(state, executor="slurm", workers=2, slurm=per_cpu)[-3:] == [
        "--default-resources",
        "mem_mb_per_cpu=4000",
        "runtime=60",
    ]
    for bad, match in [
        ({"slurm_partition": "cpu_p"}, "runtime and exactly one of mem_mb and mem_mb_per_cpu"),
        ({"mem_mb": 1000, "mem_mb_per_cpu": 1000, "runtime": 60}, "exactly one of mem_mb and mem_mb_per_cpu"),
        ({"mem_mb": "16G", "runtime": 60}, "mem_mb must be a positive int of MB, not '16G'"),
        ({"mem_mb": True, "runtime": 60}, "mem_mb must be a positive int of MB, not True"),
        ({"mem_mb": 1000, "runtime": 60, "fit_stain": {"mem_mb": 6.4e4}}, r"fit_stain.*mem_mb must be a positive int"),
        ({"mem_mb": 1000, "runtime": 60, "fit_stain": 64000}, "fit_stain.*must be a mapping"),
    ]:
        with pytest.raises(ValueError, match=match):
            _snakemake.command(state, executor="slurm", workers=8, slurm=bad)
    with pytest.raises(ValueError, match="slurm"):
        _snakemake.command(state, executor="local", workers=1, slurm=slurm)


def test_snakefile_checks(cohort, tmp_path):
    dest = tmp_path / "c.zarr"
    chain = default_chain(0.5)
    plan, _ = _materialize._prepare(
        cohort, dest, adapters=chain, role="he", roles=ROLES, mpp_native=MPP, obs_columns=(), allow_mixed_grids=False
    )
    slurm = {"mem_mb": 16000, "runtime": "2h", "fit_stain": {"mem_mb": 64000, "runtime": "4h"}}
    state = _snakemake.write_workflow(plan, chain, executor="slurm", slurm=slurm)
    snakefile = (state / "Snakefile").read_text()
    fit_rule = snakefile[snakefile.index("rule fit_stain:") : snakefile.index("rule dataset_uidA:")]
    assert "    resources:\n        mem_mb=64000,\n        runtime='4h',\n" in fit_rule
    assert snakefile.count("resources:") == 1
    for rule in ("validate", "fit_stain", "dataset_uidA", "dataset_uidB"):
        assert repr(str(state / "logs" / f"{rule}.log")) in snakefile
    assert snakefile.count("> {log:q} 2>&1") == 4
    # snakemake parses the rule resources (a local dry run of the same Snakefile)
    out = subprocess.run(_snakemake.command(state, executor="local", workers=1, dry_run=True), **_capture())
    assert _job_stats(out.stdout + out.stderr)["total"] == 4

    def with_member(uid, path):
        members = {**plan["members"], uid: {**plan["members"]["uidA"], "path": path}}
        return {**plan, "members": members, "todo": [*plan["todo"], uid]}

    source = plan["members"]["uidA"]["path"]
    for uid in ("uid{A}", "a/b", " uidA", "uid\nA"):
        with pytest.raises(ValueError, match="snakemake reads specially"):
            _snakemake.write_workflow(with_member(uid, source), chain)
    with pytest.raises(ValueError, match="store .*nope.zarr does not exist"):
        _snakemake.write_workflow(with_member("uidX", str(tmp_path / "nope.zarr")), chain)
    relative = with_member("uidX", "uidA.zarr")
    with pytest.raises(ValueError, match="executor='slurm' the store must be an absolute path"):
        _snakemake.write_workflow(relative, chain, executor="slurm", slurm=slurm)


def test_failed_member_names_its_log(cohort, tmp_path, monkeypatch):
    # the jobs are other processes: a sitecustomize makes validate_dataset fail for uidB's staged dataset there
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('SDDB_TEST_FAIL_UID'):\n"
        "    import sddb.loader._materialize as m\n"
        "    real, uid = m.validate_dataset, os.environ['SDDB_TEST_FAIL_UID']\n"
        "    def fail(group, **kw):\n"
        "        if f'/{uid}/stage.zarr' in str(group.store):\n"
        "            return ['synthetic issue']\n"
        "        return real(group, **kw)\n"
        "    m.validate_dataset = fail\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(hook))
    monkeypatch.setenv("SDDB_TEST_FAIL_UID", "uidB")
    dest, state = tmp_path / "c.zarr", tmp_path / "c.zarr.sddb"
    chain = [Rasterize(0.5), Centroids()]
    with pytest.raises(RuntimeError) as raised:
        materialize(cohort, dest, adapters=chain, role="he", roles=ROLES, mpp_native=MPP, runner="snakemake")
    log = state / "logs" / "dataset_uidB.log"
    message = str(raised.value)
    assert f"rule dataset_uidB failed, its log is {log}" in message
    assert "dataset_1 (uidB): synthetic issue" in message
    assert "dataset_1 (uidB): synthetic issue" in log.read_text()
    assert not (dest / "dataset_1").exists()
    assert not (state / "done" / "uidB").exists()


def test_rerun_writes_only_the_new_member(tmp_path):
    paths = {uid: write_member(tmp_path, uid, seed=i)[0] for i, uid in enumerate(("uidA", "uidB", "uidC"))}
    dest, state = tmp_path / "c.zarr", tmp_path / "c.zarr.sddb"
    kwargs = {"adapters": [Rasterize(0.5), Centroids()], "role": "he", "roles": ROLES, "mpp_native": MPP}
    materialize(FakeCohort(FakeMember(u, paths[u]) for u in ("uidA", "uidB")), dest, runner="snakemake", **kwargs)
    before = {u: (state / "done" / u).stat().st_mtime_ns for u in ("uidA", "uidB")}

    materialize(FakeCohort(FakeMember(u, p) for u, p in paths.items()), dest, runner="snakemake", **kwargs)
    log = (state / "snakemake.log").read_text()
    assert log.count("rule dataset_") == 1
    assert "rule dataset_uidC:" in log
    assert {u: (state / "done" / u).stat().st_mtime_ns for u in ("uidA", "uidB")} == before
    assert validate(dest) == []
    listed = json.loads((dest / "zarr.json").read_text())["consolidated_metadata"]["metadata"]
    assert {"dataset_0", "dataset_1", "dataset_2", "dataset_2/images/he"} <= set(listed)
    assert _contract(dest, "dataset_2").uid == "uidC"


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


def test_rerun_after_a_failed_member_keeps_the_stain_fit(cohort, tmp_path, monkeypatch):
    # the fit is the slow step (48 min on four Atera slides): a rerun with the same chain and members reuses it
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('SDDB_TEST_FAIL_UID'):\n"
        "    import sddb.loader._materialize as m\n"
        "    real, uid = m.validate_dataset, os.environ['SDDB_TEST_FAIL_UID']\n"
        "    def fail(group, **kw):\n"
        "        if f'/{uid}/stage.zarr' in str(group.store):\n"
        "            return ['synthetic issue']\n"
        "        return real(group, **kw)\n"
        "    m.validate_dataset = fail\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(hook))
    monkeypatch.setenv("SDDB_TEST_FAIL_UID", "uidB")
    dest, state = tmp_path / "c.zarr", tmp_path / "c.zarr.sddb"
    kwargs = {"role": "he", "roles": ROLES, "mpp_native": MPP, "runner": "snakemake"}
    with pytest.raises(RuntimeError):
        materialize(cohort, dest, adapters=default_chain(0.5), **kwargs)
    fitted = (state / "stain.json").stat().st_mtime_ns

    monkeypatch.delenv("SDDB_TEST_FAIL_UID")
    materialize(cohort, dest, adapters=default_chain(0.5), **kwargs)
    assert (state / "stain.json").stat().st_mtime_ns == fitted
    assert "rule fit_stain:" not in (state / "snakemake.log").read_text()
    assert validate(dest) == []

    # another chain is another fit
    _snakemake.write_workflow(yaml.safe_load((state / "config.yaml").read_text()), default_chain(0.25))
    assert not (state / "stain.json").exists()


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


def test_jobs_read_x_lazily_and_write_it_whole(tmp_path):
    """The jobs keep X as dask until the write (_materialize._read_store), and write what the in-process run writes."""
    import dask.array as da
    import spatialdata as sd
    from scipy import sparse

    members = []
    for uid, seed in (("uidA", 0), ("uidB", 1)):
        path = write_member(tmp_path, uid, seed=seed, extra_gene=uid == "uidB")[0]
        sdata = sd.read_zarr(path)
        table = sdata.tables["table"]
        table.X = sparse.csr_matrix(table.X)  # Atera's X is CSR
        sparse_path = tmp_path / f"{uid}_csr.zarr"
        sdata.write(sparse_path)
        members.append(FakeMember(uid, sparse_path))
        lazy = _materialize._read_store(sparse_path).tables["table"]
        assert isinstance(lazy.X, da.Array)
        assert sparse.issparse(lazy.X.compute())
        np.testing.assert_array_equal(lazy.X.compute().toarray(), table.X.toarray())
    cohort = FakeCohort(members)
    kwargs = {"adapters": [Rasterize(0.5), Centroids()], "role": "he", "roles": ROLES, "mpp_native": MPP}
    local = materialize(cohort, tmp_path / "local.zarr", **kwargs)
    snake = materialize(cohort, tmp_path / "snake.zarr", runner="snakemake", **kwargs)
    assert validate(snake) == []
    for key in ("dataset_0", "dataset_1"):
        a, b = zarr.open_group(local / key, mode="r"), zarr.open_group(snake / key, mode="r")
        assert a["X"].attrs["encoding-type"] == b["X"].attrs["encoding-type"] == "csr_matrix"
        for part in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(a["X"][part][...], b["X"][part][...], err_msg=f"{key} X/{part}")


def test_fit_key_changes_with_roles_and_mpp_native():
    config = {"adapters": [], "roles": dict(ROLES), "members": {"uidA": {"mpp_native": 0.25, "path": "a.zarr"}}}
    key = _snakemake.fit_key(config)
    assert _snakemake.fit_key({**config, "members": {"uidA": {"mpp_native": 0.25, "path": None}}}) == key
    assert _snakemake.fit_key({**config, "members": {"uidA": {"mpp_native": 0.5, "path": "a.zarr"}}}) != key
    assert _snakemake.fit_key({**config, "roles": {**ROLES, "image": "^images/other$"}}) != key
