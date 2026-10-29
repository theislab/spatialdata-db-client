import sys
from importlib.metadata import version as _version
from pathlib import Path

from sddb._catalog_schema import CATALOG_COLUMNS, FACETS, RANGE_COLUMNS

sys.path.insert(0, str(Path(__file__).parent / "_ext"))
from catalog_stats import SITE_URL, fill_code_examples, load_stats, warn_if_unavailable, with_defaults
from cli_docs import cli_markdown
from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx


def _write_catalog_columns() -> None:
    rows = ["| Column | Dtype | Facet | Range |", "|---|---|---|---|"]
    for col, dtype in CATALOG_COLUMNS.items():
        rows.append(
            f"| `{col}` | `{dtype}` | {'yes' if col in FACETS else ''} | {'yes' if col in RANGE_COLUMNS else ''} |"
        )
    (Path(__file__).parent / "reference" / "_catalog_columns.md").write_text("\n".join(rows) + "\n")


_write_catalog_columns()
(Path(__file__).parent / "reference" / "_cli.md").write_text(cli_markdown())

project = "spatialdata-db"
author = "Theis Lab"
copyright = "2026, Theis Lab"
release = _version("spatialdata-db")

extensions = ["scverse_doc"]
source_repository = "https://github.com/theislab/spatialdata-db-client"
exclude_patterns = [
    *DEFAULTS.exclude_patterns,
    "design",
    "superpowers",
    "**/.pixi",
    "reference/_catalog_columns.md",
    "reference/_cli.md",
]
smartquotes = False  # keep straight quotes, and `--flag` in the CLI reference instead of an en dash

copybutton_prompt_text = "$ "
suppress_warnings = ["sddb.catalog"]
myst_enable_extensions = [*DEFAULTS.myst_enable_extensions, "substitution"]
nb_execution_mode = "off"  # guides are executed on the HPC by docs/execute_notebooks.sh, never at build time
myst_substitutions = with_defaults(load_stats())


def _drop_scverse_links(app):
    """Remove the scverse Discourse icon that the scverse theme appends to ``icon_links``."""
    options = app.config.html_theme_options
    options["icon_links"] = [link for link in options.get("icon_links", []) if "scverse.org" not in link["url"]]


def setup(app):
    """Emit the deferred catalog warning once ``suppress_warnings`` is active."""
    app.connect("config-inited", warn_if_unavailable)
    app.connect("source-read", fill_code_examples)
    app.connect("builder-inited", _drop_scverse_links, priority=900)  # after the theme appends it (500)


intersphinx_mapping = intersphinx()
ogp_social_cards = {"enable": False}  # needs matplotlib, and the pages have no card artwork
xref_overrides = {"pandas.core.series.Series": "pandas.Series"}
nitpick_ignore = [
    # Internal classes that appear in signatures but are not exported from `sddb`.
    ("py:class", "sddb.dataset.Source"),
    ("py:class", "sddb.manifest.ManifestEntry"),
    ("py:class", "sddb.objectmeta.ObjectMeta"),
    ("py:class", "sddb.genes.GeneIndex"),
    ("py:class", "sddb.features.FeatureIndex"),
    # The docstrings of `count` and `index`, inherited from `collections.abc.Sequence`, read as return types.
    ("py:class", "integer -- return number of occurrences of value"),
    ("py:class", "integer -- return first index of value."),
]

html_logo = "_static/logo.svg"  # og:image and the fallback; the sidebar shows the wordmarks below
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["css/sddb.css"]
html_theme_options = {
    "accent": "#4fc3ea",  # the website's cyan
    "secondary_accent": "#4fc3ea",  # the sidebar captions, in the same cyan
    "logo": {
        "image_light": "_static/wordmark.svg",
        "image_dark": "_static/wordmark-dark.svg",
        "alt_text": "SpatialData-DB - Home",
    },
    "icon_links": [
        {"name": "Browse the DB", "url": SITE_URL, "icon": "fa-solid fa-globe"},
        {"name": "Lamin", "url": "https://lamin.ai/scverse/spatialdata-db", "icon": "fa-solid fa-database"},
    ],
    # Not a scverse package (yet): no scverse menu, announcement, footer or Discourse link.
    "article_header_end": ["theme-switcher", "navbar-icon-links", "toggle-secondary-sidebar"],
    "announcement": "",
    "footer_start": ["copyright"],
}
