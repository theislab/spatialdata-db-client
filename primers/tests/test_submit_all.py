import importlib.util
from pathlib import Path

import yaml


def _mod():
    """Import hpc/submit_all.py, which is not part of the package."""
    spec = importlib.util.spec_from_file_location(
        "submit_all", Path(__file__).resolve().parents[1] / "hpc" / "submit_all.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cohorts(tmp_path):
    path = tmp_path / "cohorts.yaml"
    cohorts = [{"slug": "a"}, {"slug": "b", "reserve": True}, {"slug": "c"}, {"slug": "d"}]
    path.write_text(yaml.safe_dump({"cohorts": cohorts}))
    return path


def test_slugs_skip_reserve_and_honour_only(tmp_path):
    """Reserve cohorts are left out, and ``only`` limits the list in the file's order."""
    mod = _mod()
    assert mod.slugs(_cohorts(tmp_path)) == ["a", "c", "d"]
    assert mod.slugs(_cohorts(tmp_path), only=["d", "a"]) == ["a", "d"]


def test_command_chains_on_the_previous_job():
    """The first job has no dependency, and each later one waits for the previous job to end."""
    mod = _mod()
    assert not any(a.startswith("--dependency") for a in mod.command("a"))
    second = mod.command("c", after="123")
    assert "--dependency=afterany:123" in second
    assert "--export=ALL,SLUG=c" in second
    assert second[-1].endswith("run_primer.sbatch")
