import sys
from importlib.metadata import version as _version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))
from catalog_stats import load_stats, with_defaults  # noqa: E402

project = "spatialdata-db"
author = "Tim Treis and the spatialdata-db authors"
copyright = "2026, Tim Treis and the spatialdata-db authors"
release = _version("spatialdata-db")

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_click",
]
exclude_patterns = ["_build", "design", "superpowers"]
templates_path = ["_templates"]

nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "substitution", "deflist"]
myst_substitutions = with_defaults(load_stats())

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
        {"name": "Browse the DB", "url": "https://spatialdata-db.com", "icon": "fa-solid fa-globe"},
        {
            "name": "GitHub",
            "url": "https://github.com/theislab/spatialdata-db-client",
            "icon": "fa-brands fa-github",
        },
        {"name": "Lamin", "url": "https://lamin.ai/scverse/spatialdata-db", "icon": "fa-solid fa-database"},
    ],
}
