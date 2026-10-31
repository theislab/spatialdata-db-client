"""Cohort citations: the published citations.bib entries of a cohort's studies, plus fallback entries for the rest."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from sddb._cache import fetch_catalog
from sddb.catalog import SITE_URL
from sddb.genes import sibling_url

if TYPE_CHECKING:
    from sddb.cohort import SpatialDataCohort

_DOI_FIELD = re.compile(r"\bdoi\s*=\s*[{\"]\s*([^}\"]+?)\s*[}\"]", re.IGNORECASE)
_NOTE_FIELD = re.compile(r"\bnote\s*=\s*\{", re.IGNORECASE)
_SHOWN = 5  # keys listed in a merged entry's note before the rest become a count
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


def _doi(entry: str) -> str | None:
    """The DOI field of a BibTeX entry, lower case and without a resolver prefix, or None."""
    match = _DOI_FIELD.search(entry)
    if not match:
        return None
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", match.group(1).strip(), flags=re.IGNORECASE)
    return doi.lower() or None


def _add_note(entry: str, text: str) -> str:
    """Append ``text`` to the entry's ``note`` field, or add a ``note`` field before the closing brace."""
    if match := _NOTE_FIELD.search(entry):
        depth, j = 1, match.end()
        while j < len(entry) and depth:
            depth += {"{": 1, "}": -1}.get(entry[j], 0)
            j += 1
        return f"{entry[: j - 1]}; {text}{entry[j - 1 :]}"
    body = entry[: entry.rindex("}")].rstrip().rstrip(",")
    return f"{body},\n  note = {{{text}}}\n}}"


def _merge_by_doi(entries: list[tuple[str, str]]) -> list[str]:
    """Merge ``(key, entry)`` pairs that share a DOI into the first one; its note lists every key it covers.

    Entries without a DOI are kept as they are, and the order of the first entry of each DOI is kept.
    """
    by_doi: dict[str, list[str]] = {}
    for key, entry in entries:
        if doi := _doi(entry):
            by_doi.setdefault(doi, []).append(key)
    out = []
    for key, entry in entries:
        keys = by_doi.get(_doi(entry) or "", [key])
        if keys[0] != key:
            continue  # merged into the first entry of its DOI
        if len(keys) > 1:
            shown = ", ".join(_tex(k) for k in keys[:_SHOWN])
            more = f" and {len(keys) - _SHOWN} more" if len(keys) > _SHOWN else ""
            entry = _add_note(entry, f"covers samples: {shown}{more}")
        out.append(entry)
    return out


def citation_entries(results: SpatialDataCohort, *, bib_url: str | None = None) -> tuple[str, int, int]:
    """Return ``(BibTeX text, n_published, n_fallback)`` that cite every study in ``results``.

    ``bib_url`` defaults to ``citations.bib`` next to the catalog. A study whose ``study_id`` is a key
    in that file gets its published entry. Every other study, and every dataset without a
    ``study_id``, gets a ``@misc`` fallback entry keyed by the ``study_id`` (or the uid). It cites the
    dataset page on the website and carries the title, DOI, year, uid and license from the catalog.
    The datasets of one study share one entry.

    The published file has one entry per study, and a GEO study is one sample, so a paper with many
    samples appears many times. Published entries that share a DOI are merged into one: the first
    (by key) keeps its key and fields, and its ``note`` gains "covers samples: <key>, ..." with every
    key it stands for (the first five, then a count). Fallback entries that share a DOI merge the
    same way. ``n_published`` and ``n_fallback`` count the entries after merging.

    One warning reports the counts when there are fallback entries, because those entries should be
    checked before you publish. An empty cohort raises ``ValueError``.
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
    fallback_keys = [(str(k), _fallback_entry(str(k), g)) for k, g in rest.groupby(keys[rest.index])]
    merged = _merge_by_doi([(k, entries[k]) for k in published])
    fallback = _merge_by_doi(fallback_keys)
    if fallback:
        n_studies = len(published) + len(fallback_keys)
        warnings.warn(
            f"{len(published)} of {n_studies} studies have a published citation, in {len(merged)} entries "
            f"after merging those that share a DOI; added {len(fallback)} fallback entries that cite the "
            "dataset page; check them before publishing",
            stacklevel=2,
        )
    out = "".join(e + "\n\n" for e in [*merged, *fallback])
    return out, len(merged), len(fallback)


def citations_text(results: SpatialDataCohort, *, bib_url: str | None = None) -> tuple[str, int]:
    """Return ``(BibTeX text, entry count)`` for the studies in ``results``; see :func:`citation_entries`."""
    text, n_published, n_fallback = citation_entries(results, bib_url=bib_url)
    return text, n_published + n_fallback


def write_citations(results: SpatialDataCohort, path: str | Path, *, bib_url: str | None = None) -> Path:
    """Write a BibTeX entry for every study in ``results`` to ``path``; see :func:`citation_entries`."""
    out = Path(path)
    out.write_text(citation_entries(results, bib_url=bib_url)[0], encoding="utf-8")
    return out
