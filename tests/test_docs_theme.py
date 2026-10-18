import re
from pathlib import Path

CSS = Path(__file__).parent.parent / "docs" / "_static" / "css" / "sddb.css"


def _block(css: str, selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert m, f"no {selector} block"
    return m.group(1)


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_css_has_both_modes():
    css = CSS.read_text()
    for mode, primary, bg in (("dark", "#4fc3ea", "#0b0e1a"), ("light", "#1484c0", "#f7f8fb")):
        block = _block(css, f'html[data-theme="{mode}"]')
        assert f"--pst-color-primary: {primary};" in block
        assert f"--pst-color-background: {bg};" in block


def test_dark_code_contrast():
    assert _contrast("#e8eaf2", "#1b2238") >= 4.5
