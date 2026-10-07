"""Cohort citations: filter a published citations.bib to the studies in a SpatialDataCohort."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from sddb._cache import fetch_catalog
from sddb.genes import sibling_url

if TYPE_CHECKING:
    from sddb.cohort import SpatialDataCohort

DEFAULT_BIB_URL = "https://github.com/theislab/spatialdata-db-client/releases/latest/download/citations.bib"


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


def citations_text(results: SpatialDataCohort, *, bib_url: str | None = None) -> tuple[str, int]:
    """Return ``(BibTeX text, entry count)`` for the studies in ``results``.

    ``bib_url`` defaults to ``citations.bib`` next to the catalog. Studies without an entry are
    skipped; a warning is emitted if the catalog has no ``study_id`` or nothing matches.
    """
    df = results.to_df()
    ids = set(df["study_id"].dropna().astype(str)) if "study_id" in df.columns else set()
    src = results._source
    url = bib_url or (sibling_url(src.url, "citations.bib") if src.url else DEFAULT_BIB_URL)
    text = fetch_catalog(url, cache_dir=src.cache_dir, refresh=src.refresh).read_text(encoding="utf-8")
    entries = parse_bibtex(text)
    keys = [k for k in sorted(ids) if k in entries]
    if "study_id" not in df.columns:
        warnings.warn("catalog has no 'study_id' column; no citations", stacklevel=2)
    elif not keys:
        warnings.warn("no cite-keys match the cohort's study_id values; no citations", stacklevel=2)
    return "".join(entries[k] + "\n\n" for k in keys), len(keys)


def write_citations(results: SpatialDataCohort, path: str | Path, *, bib_url: str | None = None) -> Path:
    """Write the BibTeX entries whose cite-key is a ``study_id`` in ``results`` to ``path``."""
    out = Path(path)
    out.write_text(citations_text(results, bib_url=bib_url)[0], encoding="utf-8")
    return out
