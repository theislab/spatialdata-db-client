"""Web hero figures for the Atera collection page, from the primer's outputs.

    python export_hero.py ATERA_OUT DEST      -> DEST/*.webp (<= 1600 px wide, < 400 KB, white background) and DEST/primer.json
"""
import io
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CREDIT = "Data: 10x Genomics, CC BY 4.0."
FIGS = [  # (source png, web name, alt text, caption), in the page's order
    ("fig_qc.png", "atera-qc.webp", "Violin plots of counts and genes per cell for each Atera section",
     f"Counts and detected genes per cell in each of the six Atera sections, after the same quality filters. {CREDIT}"),
    ("fig_umap_dataset.png", "atera-umap-dataset.webp", "UMAP of all cells coloured by section, before and after Harmony",
     f"All cells from the six sections in one UMAP, before and after Harmony integration across sections. {CREDIT}"),
    ("fig_cell_types.png", "atera-cell-types.webp", "UMAP coloured by broad cell type, with the cell-type share per tissue",
     f"Broad cell types from marker-gene scores per cluster, an automated and cursory annotation, and their share in each tissue. {CREDIT}"),
    ("fig_cell_types_space.png", "atera-cell-types-space.webp", "Each Atera section with its cells coloured by broad cell type",
     f"The same automated, cursory cell types drawn on each section with spatialdata-plot. {CREDIT}"),
]


def squeeze_blank_rows(im, keep=30):
    """Shorten every run of all-white rows (e.g. between the last panel row and the credit footer) to `keep` px."""
    a = np.asarray(im)
    blank = (a > 250).all(axis=(1, 2))
    rows, run = [], 0
    for i, b in enumerate(blank):
        run = run + 1 if b else 0
        if run <= keep:
            rows.append(i)
    return Image.fromarray(a[rows])


def save_webp(src, dest, max_w=1600, max_kb=400):
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    im = Image.alpha_composite(bg, im).convert("RGB")
    im = squeeze_blank_rows(im)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    for q in (90, 80, 70, 60, 50, 40):  # highest quality under the size cap
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=q, method=6)
        if buf.tell() < max_kb * 1024:
            break
    else:
        raise ValueError(f"{src}: still {buf.tell() // 1024} KB at quality 40")
    dest.write_bytes(buf.getvalue())
    return im.width, im.height, buf.tell()


def main(src_dir, dest):
    src_dir, dest = Path(src_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    figures = []
    for src, name, alt, cap in FIGS:
        w, h, size = save_webp(src_dir / src, dest / name)
        figures.append({"src": name, "alt": alt, "caption": cap, "width": w, "height": h})
        print(name, w, h, size // 1024, "KB")
    summary = ("spatialdata-db holds all six 10x Atera whole-transcriptome sections, about 2.3 million cells, in one format. "
               "Here the same quality control, Harmony integration and a cursory cell-type annotation run on all of them on one GPU. "
               + CREDIT)
    # quality control per section (qc_per_dataset.csv of the same run): low-quality cells are counted on the page
    qc = list(csv.DictReader((src_dir / "qc_per_dataset.csv").open()))
    qc = [{"section": f"{r['tissue']} ({r['preservation']})", "cells": int(r["cells"]), "low_quality": int(r["low_quality"]),
           "cells_kept": int(r["cells_kept"]), "median_counts": float(r["median_counts"])} for r in qc]
    # the page story: the numbered paragraphs of STORY_DRAFT.md next to this script, in order
    story = [m.group(1) for m in re.finditer(r"^\d+\. (.+)$", (Path(__file__).with_name("STORY_DRAFT.md")).read_text(), re.M)]
    (dest / "primer.json").write_text(json.dumps({"summary": summary, "figures": figures, "qc": qc, "story": story}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
