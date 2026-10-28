"""Build the thumbnail grids on the docs landing page.

The grids show one dataset for each of the 12 largest technologies, picked the way the website's front page
picks them: group the catalog by its ``assay`` display family, take the 12 largest families (ties broken by
name) and, in each, the first row with a Vitessce view (else the first row). The tiles are the website's card
thumbnails, labelled like its cards. A family whose pick has no thumbnail, or a blank one, falls back to its
next row with a usable one.

It writes two files with the same tiles: ``docs/_static/landing-mosaic.webp`` (4 x 3, for wide screens) and
``docs/_static/landing-mosaic-narrow.webp`` (2 x 6, for phones). The labels use Inter Semibold, the docs
theme's typeface, fetched from a pinned npm package, so a refresh looks the same on every machine.

The docs build does not run this script, and its output is committed. To refresh it, run from the repository
root (needs network access to the catalog, the website and jsDelivr)::

    uv run --with "pillow>=10.1" python docs/scripts/landing_mosaic.py
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

from sddb import Catalog

SITE_URL = "https://spatialdata-db-web.pages.dev"
STATIC = Path(__file__).parent.parent / "_static"
LAYOUTS = {"landing-mosaic.webp": 4, "landing-mosaic-narrow.webp": 2}  # file name: columns
N_TILES = 12
GAP, RADIUS = 16, 14
TILE_W = (1600 - 3 * GAP) // 4  # the wide grid is 1600 px, the narrow one 2 tiles of the same size
TILE_H = TILE_W * 3 // 4  # the card thumbnails are 4:3
FONT_URL = "https://cdn.jsdelivr.net/npm/@fontsource/inter@5.0.8/files/inter-latin-600-normal.woff"
FONT_SIZE = 30  # about 13 px once a grid is scaled to the page; a label too long for one line wraps
PAD = 22
SCRIM = 0.72  # opacity of the black band behind a label: white text on it passes 4.5:1 even on a white tile
FADE = 40  # height of the gradient above the band


def display_assay(assay: str) -> str:
    """The website's ``displayAssay`` (src/lib/inventory-core.mjs)."""
    return "Visium" if assay == "Visium Spatial Gene Expression" else assay


def get(url: str) -> bytes:
    """Fetch ``url``; Cloudflare refuses urllib's default User-Agent."""
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sddb-docs"}), timeout=30) as resp:
        return resp.read()


def thumbnail(slug: str) -> Image.Image | None:
    """The website's card thumbnail for ``slug``, or None if it has none or it is blank (nearly one colour)."""
    try:
        img = Image.open(io.BytesIO(get(f"{SITE_URL}/thumbnails/{slug}_card.webp"))).convert("RGB")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    return None if ImageStat.Stat(img.convert("L")).stddev[0] < 3 else img


def picks(n: int = N_TILES) -> list[tuple[str, Image.Image]]:
    """One ``(family, thumbnail)`` for each of the ``n`` largest technology families, by count and then name."""
    df = Catalog().query().to_df()
    df["family"] = df["assay"].astype(str).map(display_assay)
    counts = df["family"].value_counts().rename_axis("family").reset_index(name="n")
    families = counts.sort_values(["n", "family"], ascending=[False, True])["family"][:n]
    out = []
    for fam in families:
        rows = df[df["family"] == fam]
        rows = rows.iloc[rows["vitessce_url"].isna().argsort(kind="stable")]  # rows with a Vitessce view first
        img = next((im for im in map(thumbnail, rows["slug"]) if im is not None), None)
        if img is None:
            raise RuntimeError(f"no thumbnail for any {fam} dataset")
        out.append((fam, img))
    return out


def font() -> ImageFont.FreeTypeFont:
    """Inter Semibold at ``FONT_SIZE``."""
    return ImageFont.truetype(io.BytesIO(get(FONT_URL)), FONT_SIZE)


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
    """One rounded tile with the label bottom left on a dark band that fades out upwards."""
    out = cover(img).convert("RGBA")
    label = wrap(label, typeface)
    text = {"font": typeface, "anchor": "ld", "spacing": 6}
    x, y = PAD, TILE_H - PAD
    top = ImageDraw.Draw(out).multiline_textbbox((x, y), label, **text)[1]
    band_h = TILE_H - top + PAD
    alpha = Image.new("L", (TILE_W, band_h + FADE), round(255 * SCRIM))
    fade = Image.linear_gradient("L").resize((TILE_W, FADE)).point(lambda v: v * SCRIM)
    alpha.paste(fade, (0, 0))
    scrim = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    scrim.putalpha(alpha)
    out.alpha_composite(scrim, (0, TILE_H - alpha.height))
    ImageDraw.Draw(out).multiline_text((x, y), label, fill="white", **text)
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, TILE_W - 1, TILE_H - 1), RADIUS, fill=255)
    out.putalpha(mask)
    return out


def grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    """Lay ``tiles`` out in ``cols`` columns with ``GAP`` between them, on a transparent canvas."""
    rows = -(-len(tiles) // cols)
    size = (cols * TILE_W + (cols - 1) * GAP, rows * TILE_H + (rows - 1) * GAP)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.alpha_composite(t, (c * (TILE_W + GAP), r * (TILE_H + GAP)))
    return canvas


def main() -> None:
    """Make the tiles once and write both grids."""
    typeface = font()
    picked = picks()
    tiles = [tile(img, fam, typeface) for fam, img in picked]
    print("tiles:", ", ".join(fam for fam, _ in picked))
    for name, cols in LAYOUTS.items():
        out = STATIC / name
        grid(tiles, cols).save(out, "WEBP", quality=72, method=6)
        print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
