"""Cohort citations: the published citations.bib entries of a cohort's studies, plus fallback entries for the rest."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from sddb._cache import fetch_catalog
from sddb.catalog import SITE_URL
from sddb.genes import sibling_url

if TYPE_CHECKING:
    from sddb.cohort import SpatialDataCohort

DEFAULT_BIB_URL = "https://github.com/theislab/spatialdata-db-client/releases/download/catalog-latest/citations.bib"


def parse_bibtex(text: str) -> dict[str, str]:
    """Split BibTeX text into ``{cite_key: raw_entry}`` with a brace-matching scan.

    Entries start at an ``@`` outside any braces; ``@comment``/``@string``/``@preamble`` are skipped.
    """
    out: dict[str, str] = {}
    i, n = 0, len(text)
    while i < n:
        at = text.find("@", i)
        if at < 0:
            break
        brace = text.find("{", at)
        if brace < 0:
            break
        depth, j = 0, brace
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        kind = text[at + 1 : brace].strip().lower()
        body = text[brace + 1 : j]
        key = body.split(",", 1)[0].strip()
        if kind not in {"comment", "string", "preamble"} and key:
            out[key] = text[at : j + 1]
        i = j + 1
    return out


def _tex(value: object) -> str:
    """Make a catalog string safe inside a BibTeX ``{...}`` value: drop braces, escape LaTeX specials."""
    text = str(value).replace("\\", "").replace("{", "").replace("}", "")
    for ch in "&%$#_":
        text = text.replace(ch, "\\" + ch)
    return " ".join(text.split())


def _first(rows: pd.DataFrame, col: str) -> str | None:
    """The first non-empty value of ``col`` in ``rows``, as a string, or None."""
    if col not in rows.columns:
        return None
    vals = [str(v) for v in rows[col].dropna() if str(v).strip()]
    return vals[0] if vals else None


def _fallback_entry(key: str, rows: pd.DataFrame) -> str:
    """A ``@misc`` entry citing the website page of a study (or one dataset) that has no published entry."""
    rows = rows.sort_values("uid")
    uids = rows["uid"].astype(str).tolist()
    fields = [("title", _tex(_first(rows, "title") or uids[0]))]
    if slug := _first(rows, "slug"):
        fields.append(("howpublished", f"\\url{{{SITE_URL}/datasets/{slug}/}}"))
    if doi := _first(rows, "doi"):
        fields.append(("doi", doi))
    # publication_year is set where a paper was verified; publication_date (the source's release date)
    # covers nearly every row, and published_at is the last resort.
    dates = (_first(rows, c) for c in ("publication_year", "publication_date", "published_at"))
    year = next((d[:4] for d in dates if d), None)
    if year:
        fields.append(("year", year))
    licenses = ", ".join(sorted({str(v) for v in rows["license"].dropna()})) if "license" in rows.columns else ""
    shown = ", ".join(uids[:5]) + (f" and {len(uids) - 5} more" if len(uids) > 5 else "")
    note = f"SpatialData-DB dataset{'s' if len(uids) > 1 else ''} {shown}; license {_tex(licenses or 'unknown')}"
    fields.append(("note", note))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@misc{{{key},\n{body}\n}}"


def citation_entries(results: SpatialDataCohort, *, bib_url: str | None = None) -> tuple[str, int, int]:
    """Return ``(BibTeX text, n_published, n_fallback)`` that cite every study in ``results``.

    ``bib_url`` defaults to ``citations.bib`` next to the catalog. A study whose ``study_id`` is a key
    in that file gets its published entry. Every other study, and every dataset without a
    ``study_id``, gets a ``@misc`` fallback entry keyed by the ``study_id`` (or the uid). It cites the
    dataset page on the website and carries the title, DOI, year, uid and license from the catalog.
    The datasets of one study share one entry. One warning reports the fallback count, because
    those entries should be checked before you publish. An empty cohort raises ``ValueError``.
    """
    df = results.to_df()
    if df.empty:
        raise ValueError("no datasets to cite: the cohort is empty")
    src = results._source
    url = bib_url or (sibling_url(src.url, "citations.bib") if src.url else DEFAULT_BIB_URL)
    text = fetch_catalog(url, cache_dir=src.cache_dir, refresh=src.refresh).read_text(encoding="utf-8")
    entries = parse_bibtex(text)
    study = df["study_id"] if "study_id" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    keys = study.where(study.notna(), df["uid"]).astype(str)
    published = sorted(k for k in set(keys[study.notna()]) if k in entries)
    rest = df[~keys.isin(published)]
    fallback = [_fallback_entry(str(k), g) for k, g in rest.groupby(keys[rest.index])]
    if fallback:
        warnings.warn(
            f"{len(published)} of {len(published) + len(fallback)} studies have a published citation; "
            f"added {len(fallback)} fallback entries that cite the dataset page; check them before publishing",
            stacklevel=2,
        )
    out = "".join(e + "\n\n" for e in [*(entries[k] for k in published), *fallback])
    return out, len(published), len(fallback)


def citations_text(results: SpatialDataCohort, *, bib_url: str | None = None) -> tuple[str, int]:
    """Return ``(BibTeX text, entry count)`` for the studies in ``results``; see :func:`citation_entries`."""
    text, n_published, n_fallback = citation_entries(results, bib_url=bib_url)
    return text, n_published + n_fallback


def write_citations(results: SpatialDataCohort, path: str | Path, *, bib_url: str | None = None) -> Path:
    """Write a BibTeX entry for every study in ``results`` to ``path``; see :func:`citation_entries`."""
    out = Path(path)
    out.write_text(citation_entries(results, bib_url=bib_url)[0], encoding="utf-8")
    return out
