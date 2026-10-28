"""Build ``docs/_static/landing-mosaic.webp``, the thumbnail grid on the docs landing page.

The grid shows one dataset per technology, picked the way the website's front page picks them: group the
catalog by its ``assay`` display family, take the 12 largest families and, in each, the first row with a
Vitessce view (else the first row). The tiles are the website's card thumbnails, labelled like its cards.
A family whose pick has no thumbnail, or a blank one, falls back to its next row with a usable one.

The docs build does not run this script, and its output is committed. To refresh it, run from the repository
root (needs network access to the catalog and the website)::

    uv run --with pillow python docs/scripts/landing_mosaic.py
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from sddb import Catalog

SITE_URL = "https://spatialdata-db-web.pages.dev"
OUT = Path(__file__).parent.parent / "_static" / "landing-mosaic.webp"
COLS, ROWS = 4, 3
WIDTH, GAP, RADIUS = 1600, 16, 14
TILE_W = (WIDTH - (COLS - 1) * GAP) // COLS
TILE_H = TILE_W * 3 // 4  # the card thumbnails are 4:3
FONT_SIZE = 34  # about 13 px once the figure is scaled to the docs column; a label too long for one line wraps
PAD = 22
# Semibold system fonts, first match wins; Pillow's own font is the last resort.
FONTS = [
    ("/System/Library/Fonts/SFNS.ttf", "Semibold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None),
    ("C:/Windows/Fonts/segoeuib.ttf", None),
]


def display_assay(assay: str) -> str:
    """The website's ``displayAssay`` (src/lib/inventory-core.mjs)."""
    return "Visium" if assay == "Visium Spatial Gene Expression" else assay


def thumbnail(slug: str) -> Image.Image | None:
    """The website's card thumbnail for ``slug``, or None if it has none or it is blank (nearly one colour)."""
    req = urllib.request.Request(f"{SITE_URL}/thumbnails/{slug}_card.webp", headers={"User-Agent": "sddb-docs"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    return None if ImageStat.Stat(img.convert("L")).stddev[0] < 3 else img


def picks(n: int = COLS * ROWS) -> list[tuple[str, Image.Image]]:
    """One ``(family, thumbnail)`` per technology family, largest families first."""
    df = Catalog().query().to_df()
    df["family"] = df["assay"].astype(str).map(display_assay)
    families = df["family"].value_counts(sort=True).index[:n]
    out = []
    for fam in families:
        rows = df[df["family"] == fam]
        rows = rows.iloc[rows["vitessce_url"].isna().argsort(kind="stable")]  # rows with a Vitessce view first
        img = next((im for im in map(thumbnail, rows["slug"]) if im is not None), None)
        if img is None:
            raise RuntimeError(f"no thumbnail for any {fam} dataset")
        out.append((fam, img))
    return out


def font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The first semibold system font in ``FONTS``, at ``FONT_SIZE``."""
    for path, variation in FONTS:
        if Path(path).exists():
            f = ImageFont.truetype(path, FONT_SIZE)
            if variation:
                f.set_variation_by_name(variation)
            return f
    return ImageFont.load_default(FONT_SIZE)


def wrap(label: str, typeface) -> str:
    """Break ``label`` onto as many lines as it needs to fit the tile."""
    lines = [""]
    for word in label.split():
        trial = f"{lines[-1]} {word}".strip()
        if lines[-1] and typeface.getlength(trial) > TILE_W - 2 * PAD:
            lines.append(word)
        else:
            lines[-1] = trial
    return "\n".join(lines)


def cover(img: Image.Image) -> Image.Image:
    """Scale and centre-crop ``img`` to the tile size, like CSS ``object-fit: cover``."""
    scale = max(TILE_W / img.width, TILE_H / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    left, top = (img.width - TILE_W) // 2, (img.height - TILE_H) // 2
    return img.crop((left, top, left + TILE_W, top + TILE_H))


def tile(img: Image.Image, label: str, typeface) -> Image.Image:
    """One rounded tile with the label bottom left over a dark scrim, as on the website's cards."""
    out = cover(img).convert("RGBA")
    scrim_h = int(TILE_H * 0.5)
    alpha = Image.linear_gradient("L").resize((TILE_W, scrim_h)).point(lambda v: v * 0.6)
    scrim = Image.new("RGBA", (TILE_W, scrim_h), (0, 0, 0, 0))
    scrim.putalpha(alpha)
    out.alpha_composite(scrim, (0, TILE_H - scrim_h))
    x, y = PAD, TILE_H - PAD
    shadow = Image.new("RGBA", out.size, (0, 0, 0, 0))
    label = wrap(label, typeface)
    text = {"font": typeface, "anchor": "ld", "spacing": 6}
    ImageDraw.Draw(shadow).multiline_text((x, y + 1), label, fill=(0, 0, 0, 190), **text)
    out.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(3)))
    ImageDraw.Draw(out).multiline_text((x, y), label, fill="white", **text)
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, TILE_W - 1, TILE_H - 1), RADIUS, fill=255)
    out.putalpha(mask)
    return out


def main() -> None:
    """Compose the tiles into one grid and write ``OUT``."""
    typeface = font()
    tiles = picks()
    rows = -(-len(tiles) // COLS)
    canvas = Image.new("RGBA", (WIDTH, rows * TILE_H + (rows - 1) * GAP), (0, 0, 0, 0))
    for i, (fam, img) in enumerate(tiles):
        r, c = divmod(i, COLS)
        canvas.alpha_composite(tile(img, fam, typeface), (c * (TILE_W + GAP), r * (TILE_H + GAP)))
    canvas.save(OUT, "WEBP", quality=82, method=6)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {', '.join(fam for fam, _ in tiles)}")


if __name__ == "__main__":
    main()
