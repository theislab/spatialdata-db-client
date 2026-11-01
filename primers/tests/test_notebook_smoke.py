"""Run primer.ipynb end to end on an offline fixture cohort, on the CPU."""

import ast
import importlib.util
import json
import re
import sys
from functools import partial
from pathlib import Path

import nbformat
import numpy as np
import pytest
import yaml
from conftest import member, write_spatial_store
from primers.markers import load_markers

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "primer.ipynb"
BUNDLE_FILES = {"manifest.json", "umap.parquet", "qc.parquet", "cohort.parquet", "embedding.h5ad"}
N_OBS = 150

MARKER_GENES = list(dict.fromkeys(g for genes in load_markers()["colon"]["compartments"].values() for g in genes))
SHARED = MARKER_GENES + [f"SHARED{i}" for i in range(220 - len(MARKER_GENES))]
SPOT = SHARED + [f"SPOT{i}" for i in range(80)]  # 300 genes
IMAGING = SHARED + [f"XEN{i}" for i in range(30)]  # 250 genes, 220 shared with the spot panel


def _counts(genes, per_gene, seed):
    """Poisson counts with two groups of observations: epithelial markers high in one, immune in the other."""
    rng = np.random.default_rng(seed)
    rate = np.full((N_OBS, len(genes)), float(per_gene))
    half = N_OBS // 2
    for gene in ("EPCAM", "KRT20", "CDH17", "CEACAM5"):
        rate[:half, genes.index(gene)] *= 8
    for gene in ("CD3E", "CD3D", "CD2", "CD68"):
        rate[half:, genes.index(gene)] *= 8
    return rng.poisson(rate)


def _member(uid, technology, genes, per_gene, seed, *, ensembl, lowres, **paper):
    store = partial(write_spatial_store, genes=genes, X=_counts(genes, per_gene, seed), ensembl=ensembl, lowres=lowres)
    return {**member(uid, technology, n=N_OBS, **paper), "genes": genes, "store": store}


@pytest.fixture
def fixture_cohort(tmp_path, make_catalog):
    """A catalog of 2 Visium and 2 Xenium sections from 4 papers, and a cohorts.yaml with one cohort."""
    members = [
        _member("visium1", "Visium", SPOT, 5, 1, ensembl=True, lowres=True, doi="10.1/a"),
        _member("visium2", "Visium", SPOT, 5, 2, ensembl=True, lowres=True),
        _member("xenium1", "Xenium", IMAGING, 1, 3, ensembl=False, lowres=False),
        _member("xenium2", "Xenium", IMAGING, 1, 4, ensembl=False, lowres=False),
    ]
    make_catalog(members)
    catalog = next(tmp_path.glob("build*/catalog.parquet"))
    cohorts = tmp_path / "cohorts.yaml"
    spec = {
        "slug": "fixture-colon",
        "title": "Fixture colon",
        "query": {"organism": "human", "tissue": "colon"},
        "families": ["spot", "imaging"],
        "marker_set": "colon",
        "rationale": "An offline fixture for the notebook smoke test.",
    }
    cohorts.write_text(yaml.safe_dump({"cohorts": [spec]}))
    return catalog, cohorts


def test_primer_notebook_runs_on_cpu_and_writes_the_bundle(tmp_path, fixture_cohort):
    """The notebook runs on the fixture cohort and writes every bundle file of the contract."""
    import papermill as pm

    catalog, cohorts = fixture_cohort
    out_dir = tmp_path / "bundles"
    executed = tmp_path / "primer.ipynb"
    pm.execute_notebook(
        str(NOTEBOOK),
        str(executed),
        parameters={
            "slug": "fixture-colon",
            "use_gpu": False,
            "catalog_path": str(catalog),
            "cohorts_path": str(cohorts),
            "out_dir": str(out_dir),
            "dev_note": "Development run: a test banner.",
        },
        kernel_name="python3",
        cwd=str(tmp_path),
    )
    bundle = out_dir / "fixture-colon"
    assert BUNDLE_FILES <= {p.name for p in bundle.iterdir()}

    _clean_notebook().main([str(executed)])
    nb = nbformat.read(executed, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    outputs = "\n".join(_output_text(o) for c in code_cells for o in c.get("outputs", []))
    assert "Draft reading, to be reviewed by the Lead" in "\n".join(c.source for c in nb.cells)
    assert "Cluster " in outputs
    assert "Development run: a test banner." in outputs
    for path in (str(tmp_path), sys.prefix, str(Path.home())):
        assert path not in outputs  # no absolute paths in the outputs
        assert path not in json.dumps([nb.metadata, *(c.metadata for c in nb.cells)])

    def pngs(marker):
        cell = next(c for c in code_cells if marker in c.source)
        return sum("image/png" in o.get("data", {}) for o in cell.outputs)

    assert pngs(".pl.render_shapes") >= 2  # one section per technology
    assert pngs("sc.pl.dotplot(m") >= 1


def _output_text(output):
    """The text of one output: streams, text/plain, text/markdown, text/html and error tracebacks."""
    data = output.get("data", {})
    parts = [output.get("text", ""), *(data.get(k, "") for k in ("text/plain", "text/markdown", "text/html"))]
    parts += [output.get("evalue", ""), *output.get("traceback", [])]
    return "\n".join(str(p) for p in parts)


def _clean_notebook():
    """Import scripts/clean_notebook.py, which the HPC runner applies to every executed notebook."""
    spec = importlib.util.spec_from_file_location(
        "clean_notebook", Path(__file__).resolve().parents[1] / "scripts" / "clean_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spatialdata_plot_calls(tree):
    """Calls of the form ``<expr>.pl.<name>(...)`` where ``<expr>`` is not scanpy's ``sc``."""
    for node in ast.walk(tree):
        f = getattr(node, "func", None)
        if (
            isinstance(node, ast.Call)
            and isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "pl"
            and not (isinstance(f.value.value, ast.Name) and f.value.value.id == "sc")
        ):
            yield node


def test_notebook_plots_only_with_scanpy_and_chained_spatialdata_plot():
    """No matplotlib call, and spatialdata-plot calls are parenthesised chains with one ``.pl.`` call per line."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    n_calls = 0
    for cell in (c for c in nb.cells if c.cell_type == "code"):
        assert "plt." not in cell.source
        # spatialdata-plot's method="matplotlib" is a renderer choice, not a matplotlib call
        assert not re.search(r"\bimport matplotlib|\bfrom matplotlib|\bmatplotlib\.", cell.source)
        lines = cell.source.splitlines()
        for line in lines:
            assert not re.search(r"\bax\.|\bfig\b", line), f"a matplotlib axes or figure: {line!r}"
            assert not re.search(r"=\s*[\w.\[\]\"']+\.pl\s*(#.*)?$", line), f"an aliased .pl accessor: {line!r}"
            assert not re.search(r"\.pl\s*(#.*)?$", line), f"a .pl accessor outside a call: {line!r}"
        for call in _spatialdata_plot_calls(ast.parse(cell.source)):
            n_calls += 1
            line = lines[call.func.end_lineno - 1]
            assert re.match(r"^\s*\.pl\.\w+\(", line), f"not one .pl. call per line: {line!r}"
            base = call
            while isinstance(base, ast.Call | ast.Attribute):
                base = base.func if isinstance(base, ast.Call) else base.value
            assert lines[base.lineno - 1].strip().isidentifier(), "the chain does not start on its own line"
            assert lines[base.lineno - 2].strip() == "(", "the chain is not inside parentheses"
    assert n_calls >= 2


def test_notebook_has_a_parameters_cell_and_no_outputs():
    """Papermill finds the tagged parameters cell, and the committed notebook carries no outputs."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    tagged = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(tagged) == 1
    for name in (
        "slug",
        "release",
        "use_gpu",
        "out_dir",
        "smoke",
        "primers_ref",
        "cohorts_path",
        "catalog_path",
        "dev_note",
    ):
        assert re.search(rf"^{name} = ", tagged[0].source, re.M)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
