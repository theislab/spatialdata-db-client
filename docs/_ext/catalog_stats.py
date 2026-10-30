"""Build-time catalog statistics, exposed to MyST as substitutions."""

from __future__ import annotations

import pandas as pd
from sphinx.util import logging as _logging

from sddb.catalog import SITE_URL

_log = _logging.getLogger(__name__)
_failure: list[Exception] = []  # load_stats runs while conf.py is read, before warning filters exist

# Technologies with a hand-written card on docs/intro/assays.md.
DESCRIBED = frozenset({"Visium", "VisiumHD", "Xenium", "Atera", "SpatialTranscriptomics"})
_BASE = f"{SITE_URL}/datasets"


def assay_rows(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Every technology in the catalog with its count, most common first."""
    counts = df["technology"].value_counts(sort=True)
    return sorted(((str(t), int(n)) for t, n in counts.items()), key=lambda r: (-r[1], r[0]))


def _token(tech: str) -> str:
    """Substitution-safe key: "spatial transcriptomics" -> "SpatialTranscriptomics"."""
    return "".join(w[:1].upper() + w[1:] for w in tech.split())


def _label(row: pd.Series) -> str:
    """Human link text "<organism> <tissue>", falling back to the title."""
    parts = [str(row.get(k)) for k in ("organism", "tissue") if pd.notna(row.get(k)) and row.get(k)]
    return " ".join(parts) if len(parts) == 2 else str(row["title"])


def _datasets(n: int) -> str:
    return f"{n} dataset" + ("" if n == 1 else "s")


def _examples(df: pd.DataFrame) -> dict[str, str]:
    """The website's front-page example uid (smallest dataset) and the smallest multi-member collection slug."""
    out: dict[str, str] = {}
    if "uid" not in df:
        return out
    sized = df.dropna(subset=["size_bytes"])
    sized = sized[sized["size_bytes"] > 0]
    if not sized.empty:
        row = sized.loc[sized["size_bytes"].idxmin()]
        out["small_uid"] = str(row["uid"])
        out["small_stem"] = str(row["uid"])[:16]
    if "collections" in df:
        members = df.assign(_slug=df["collections"]).explode("_slug").dropna(subset=["_slug"])
        per = members.groupby("_slug")["size_bytes"].agg(["size", "count", "sum"])
        per = per[(per["size"] > 1) & (per["count"] == per["size"])]  # every member has a size
        if not per.empty:
            out["small_collection"] = str(per["sum"].idxmin())
    return out


def catalog_stats(df: pd.DataFrame, generated_at: str | None = None) -> dict[str, str]:
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
        stats[f"example_{tok}"] = f"[{_label(row)}]({_BASE}/{row['slug']}/)"
    for tok in DESCRIBED:
        if f"n_{tok}" in stats:
            line = f"**{_datasets(int(stats[f'n_{tok}']))}** in the catalog."
            if f"example_{tok}" in stats:
                line += f" Example: {stats[f'example_{tok}']}"
            stats[f"line_{tok}"] = line
    # The website's "Technology" facet counts the `assay` column, so count the same thing here.
    n_tech = int(df["assay" if "assay" in df else "technology"].nunique())
    subject = f"The catalog of {generated_at[:10]}" if generated_at else "The catalog"
    tech = f"{n_tech} technolog" + ("y" if n_tech == 1 else "ies")
    stats["catalog_line"] = f"{subject} holds {_datasets(len(df))} from {tech}."
    stats.update(_examples(df))
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
    fallback = {
        "catalog_line": "",
        "small_uid": "<uid>",
        "small_stem": "<16-character stem>",
        "small_collection": "<slug>",
        "catalog_date": "",
        "site_url": SITE_URL,
    }
    return {**dict.fromkeys(keys, ""), **fallback, **stats}


def fill_code_examples(app, docname, source) -> None:
    """MyST does not substitute inside code fences, so fill the ``{{ small_* }}`` placeholders in the raw source."""
    for key in ("site_url", "small_uid", "small_stem", "small_collection"):
        source[0] = source[0].replace("{{ " + key + " }}", app.config.myst_substitutions[key])


def load_stats() -> dict[str, str]:
    """Stats from the live catalog; empty (never raising) when it is unreachable."""
    try:
        import sddb

        cat = sddb.Catalog()
        stats = catalog_stats(cat.to_df(), cat.generated_at)
        stats["catalog_date"] = (cat.generated_at or "")[:10]
        return stats
    except Exception as exc:
        _failure.append(exc)
        return {}


def warn_if_unavailable(app=None, config=None) -> None:
    """Log the deferred load failure; run from ``setup`` so ``suppress_warnings`` is already in effect."""
    for exc in _failure:
        _log.warning(
            "catalog stats unavailable, building docs without numbers: %r", exc, type="sddb", subtype="catalog"
        )
    _failure.clear()
