import ast
import re
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"
CSS = DOCS / "_static" / "css" / "sddb.css"


def _conf() -> dict[str, object]:
    """The literal top-level assignments of docs/conf.py (without running it, which needs the catalog)."""
    values = {}
    for node in ast.parse((DOCS / "conf.py").read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:  # literals, plus names imported from docs/_ext (stubbed)
                expr = compile(ast.Expression(node.value), "conf.py", "eval")
                values[node.targets[0].id] = eval(expr, {"__builtins__": {}, "SITE_URL": "https://site"})
            except Exception:
                values[node.targets[0].id] = None
    return values


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_scverse_theme_with_site_accent():
    conf = _conf()
    assert conf["extensions"][0] == "scverse_doc"
    assert "html_theme" not in conf  # set, it would replace the scverse theme
    assert conf["html_theme_options"]["accent"] == "#4fc3ea"
    assert "scverse-ecosystem" not in conf["html_theme_options"]["article_header_end"]  # not a scverse package (yet)
    assert "scverse-footer" not in conf["html_theme_options"]["footer_start"]


def test_theme_owns_the_palette():
    assert "--pst-color-primary" not in CSS.read_text()


def test_wordmarks_read_in_both_modes():
    logo = _conf()["html_theme_options"]["logo"]
    # scverse-doc's page backgrounds (scverse_doc/theme/__init__.py)
    for key, bg in (("image_light", "#ffffff"), ("image_dark", "#14181e")):
        svg = (DOCS / logo[key]).read_text()
        fg, muted = re.findall(r'fill="(#[0-9a-f]{6})"', svg)[-2:]
        assert "SpatialData<tspan" in svg
        assert "-DB</tspan>" in svg
        assert _contrast(fg, bg) >= 4.5, (key, fg)
        assert _contrast(muted, bg) >= 3, (key, muted)  # large text
