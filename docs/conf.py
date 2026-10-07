from importlib.metadata import version as _version

project = "spatialdata-db"
author = "Tim Treis and the spatialdata-db authors"
copyright = "2026, Tim Treis and the spatialdata-db authors"
release = _version("spatialdata-db")

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
exclude_patterns = ["_build", "design", "superpowers"]
html_theme = "sphinx_book_theme"
html_title = "spatialdata-db"
