"""Build-time catalog statistics, exposed to MyST as substitutions."""

from __future__ import annotations

import pandas as pd

_BASE = "https://spatialdata-db.com/datasets"


def assay_rows(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Every technology in the catalog with its count, most common first."""
    counts = df["technology"].value_counts(sort=True)
    return sorted(((str(t), int(n)) for t, n in counts.items()), key=lambda r: (-r[1], r[0]))


def _token(tech: str) -> str:
    """Substitution-safe key: "spatial transcriptomics" -> "SpatialTranscriptomics"."""
    return "".join(w[:1].upper() + w[1:] for w in tech.split())


def catalog_stats(df: pd.DataFrame) -> dict[str, str]:
    stats = {
        "n_datasets": str(len(df)),
        "n_studies": str(df["study_id"].nunique()),
        "n_organisms": str(df["organism"].nunique()),
    }
    passing = df[df["validation_status"] == "pass"]
    for tech, n in assay_rows(df):
        tok = _token(tech)
        stats[f"n_{tok}"] = str(n)
        sub = passing[passing["technology"] == tech]
        if len(sub):
            row = sub.loc[sub["size_bytes"].idxmin()]
            stats[f"example_{tok}"] = f"[{row['title']}]({_BASE}/{row['slug']}/)"
    return stats


def load_stats() -> dict[str, str]:
    """Stats from the live catalog; empty (never raising) when it is unreachable."""
    try:
        import sddb

        cat = sddb.Catalog()
        stats = catalog_stats(cat.to_df())
        stats["catalog_date"] = (cat.generated_at or "")[:10]
        return stats
    except Exception:
        return {}
