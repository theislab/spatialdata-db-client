import importlib.util
from pathlib import Path

import nbformat


def _mod():
    """Import scripts/clean_notebook.py, which is not part of the package."""
    spec = importlib.util.spec_from_file_location(
        "clean_notebook", Path(__file__).resolve().parents[1] / "scripts" / "clean_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_absolute_paths_leave_the_metadata(tmp_path):
    """Papermill's input and output paths and any other absolute path in the metadata are removed."""
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")])
    nb.metadata["papermill"] = {
        "input_path": "/lustre/code/primer.ipynb",
        "output_path": "/lustre/bundles/a/primer.ipynb",
        "parameters": {"slug": "a", "out_dir": "bundles"},
        "environment_variables": {},
    }
    nb.cells[0].metadata["papermill"] = {"start_time": "2026-11-01", "where": "/tmp/node/x"}
    path = tmp_path / "primer.ipynb"
    nbformat.write(nb, path)
    _mod().main([str(path)])
    text = path.read_text()
    assert "/lustre" not in text
    assert "/tmp/node" not in text
    out = nbformat.read(path, as_version=4)
    assert out.metadata["papermill"]["parameters"] == {"slug": "a", "out_dir": "bundles"}
    assert out.cells[0].metadata["papermill"]["start_time"] == "2026-11-01"
