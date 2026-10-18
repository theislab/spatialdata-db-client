"""Build-time catalog statistics, exposed to MyST as substitutions."""

from __future__ import annotations

import pandas as pd
from sphinx.util import logging as _logging

_log = _logging.getLogger(__name__)

# Technologies with a hand-written card on docs/intro/assays.md.
DESCRIBED = frozenset({"Visium", "VisiumHD", "Xenium", "Atera", "SpatialTranscriptomics"})
_BASE = "https://spatialdata-db.com/datasets"


def assay_rows(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Every technology in the catalog with its count, most common first."""
    counts = df["technology"].value_counts(sort=True)
    return sorted(((str(t), int(n)) for t, n in counts.items()), key=lambda r: (-r[1], r[0]))


def _token(tech: str) -> str:
    """Substitution-safe key: "spatial transcriptomics" -> "SpatialTranscriptomics"."""
    return "".join(w[:1].upper() + w[1:] for w in tech.split())


def _datasets(n: int) -> str:
    return f"{n} dataset" + ("" if n == 1 else "s")


def catalog_stats(df: pd.DataFrame) -> dict[str, str]:
    """Substitution values (counts, example links, ready-made prose) for the intro pages."""
    stats = {
        "n_datasets": str(len(df)),
        "n_studies": str(df["study_id"].nunique()),
        "n_organisms": str(df["organism"].nunique()),
    }
    tokens = df["technology"].map(lambda t: _token(str(t)))
    for tok, n in tokens.value_counts().items():
        stats[f"n_{tok}"] = str(int(n))
    passing = df[(df["validation_status"] == "pass")].assign(_tok=tokens)
    passing = passing.dropna(subset=["size_bytes"])
    for tok, sub in passing.groupby("_tok"):
        row = sub.loc[sub["size_bytes"].idxmin()]
        stats[f"example_{tok}"] = f"[{row['title']}]({_BASE}/{row['slug']}/)"
    for tok in DESCRIBED:
        if f"n_{tok}" in stats:
            line = f"**{_datasets(int(stats[f'n_{tok}']))}** in the catalog."
            if f"example_{tok}" in stats:
                line += f" Example: {stats[f'example_{tok}']}"
            stats[f"line_{tok}"] = line
    stats["intro_count"] = f"The catalog currently holds {_datasets(len(df))}."
    if "tier" in df:
        tiers = df["tier"].astype(object).where(df["tier"].notna(), "unassigned").value_counts()
        stats["tier_note"] = "This release holds " + ", ".join(f"{_datasets(n)} ({t})" for t, n in tiers.items()) + "."
    other: dict[str, list] = {}
    for name, n in assay_rows(df):
        if _token(name) not in DESCRIBED:
            other.setdefault(_token(name), [name, 0])[1] += n
    rest = [f"- **{name}**: {_datasets(n)}. No description yet." for name, n in other.values()]
    stats["other_assays"] = "**Other technologies**\n\n" + "\n".join(rest) if rest else ""
    return stats


def with_defaults(stats: dict[str, str]) -> dict[str, str]:
    """Blank values for every substitution the intro pages use, so a missing catalog renders cleanly."""
    keys = ["intro_count", "tier_note", "other_assays", *(f"line_{t}" for t in DESCRIBED)]
    return {**dict.fromkeys(keys, ""), **stats}


def load_stats() -> dict[str, str]:
    """Stats from the live catalog; empty (never raising) when it is unreachable."""
    try:
        import sddb

        cat = sddb.Catalog()
        stats = catalog_stats(cat.to_df())
        stats["catalog_date"] = (cat.generated_at or "")[:10]
        return stats
    except Exception as exc:
        _log.warning("catalog stats unavailable, building docs without numbers: %r", exc)
        return {}
