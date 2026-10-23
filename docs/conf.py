import sys
from importlib.metadata import version as _version
from pathlib import Path

from sddb._catalog_schema import CATALOG_COLUMNS, FACETS, RANGE_COLUMNS

sys.path.insert(0, str(Path(__file__).parent / "_ext"))
from catalog_stats import SITE_URL, load_stats, warn_if_unavailable, with_defaults
from cli_docs import cli_markdown


def _write_catalog_columns() -> None:
    rows = ["| Column | Dtype | Facet | Range |", "|---|---|---|---|"]
    for col, dtype in CATALOG_COLUMNS.items():
        rows.append(f"| `{col}` | `{dtype}` | {'yes' if col in FACETS else ''} | {'yes' if col in RANGE_COLUMNS else ''} |")
    (Path(__file__).parent / "reference" / "_catalog_columns.md").write_text("\n".join(rows) + "\n")


_write_catalog_columns()
(Path(__file__).parent / "reference" / "_cli.md").write_text(cli_markdown())

project = "spatialdata-db"
author = "Theis Lab"
copyright = "2026, Theis Lab"
release = _version("spatialdata-db")

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
]
exclude_patterns = ["_build", "design", "superpowers", "**/.pixi", "reference/_catalog_columns.md", "reference/_cli.md"]
templates_path = ["_templates"]

autosummary_generate = True
napoleon_numpy_docstring = True

nb_execution_mode = "off"
copybutton_prompt_text = "$ "
suppress_warnings = ["sddb.catalog"]
myst_enable_extensions = ["colon_fence", "substitution", "deflist"]
myst_substitutions = with_defaults(load_stats())


def setup(app):
    """Emit the deferred catalog warning once ``suppress_warnings`` is active."""
    app.connect("config-inited", warn_if_unavailable)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.scverse.org/en/stable/", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/stable/", None),
}

html_theme = "sphinx_book_theme"
html_title = "spatialdata-db"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["css/sddb.css"]
html_theme_options = {
    "repository_url": "https://github.com/theislab/spatialdata-db-client",
    "use_repository_button": True,
    "icon_links": [
        {"name": "Browse the DB", "url": SITE_URL, "icon": "fa-solid fa-globe"},
        {
            "name": "GitHub",
            "url": "https://github.com/theislab/spatialdata-db-client",
            "icon": "fa-brands fa-github",
        },
        {"name": "Lamin", "url": "https://lamin.ai/scverse/spatialdata-db", "icon": "fa-solid fa-database"},
    ],
}
