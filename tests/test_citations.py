from __future__ import annotations

import warnings

import pytest
from tests._fixtures import make_fixture_catalog, make_fixture_citations_bib

from sddb.catalog import SITE_URL
from sddb.citations import citation_entries, parse_bibtex
from sddb.cohort import SpatialDataCohort


def test_parse_nested():
    entries = parse_bibtex(make_fixture_citations_bib())
    assert set(entries) == {"Smith2023", "Lee2022", "Wong2021", "Extra1999"}
    assert "{{nested}}" in entries["Lee2022"]
    assert entries["Lee2022"].endswith("}")
    assert "{The Consortium}" in entries["Smith2023"]


def test_results_citations(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    res = SpatialDataCohort(make_fixture_catalog().iloc[:2])  # Smith2023 only
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # every member has a published entry: no fallback warning
        out = res.citations(tmp_path / "refs.bib", bib_url=str(bib))
    text = out.read_text()
    assert "Smith2023" in text
    assert "Lee2022" not in text
    assert "Extra1999" not in text
    assert list(parse_bibtex(text)) == ["Smith2023"]
    assert citation_entries(res, bib_url=str(bib))[1:] == (1, 0)


def _with_pages(df):
    """Add the website columns the fallback entries read; the base fixture leaves them out."""
    df = df.copy()
    df["slug"] = [f"page-{u}" for u in df["uid"]]
    df["title"] = [f"Dataset {u} 50% & co" for u in df["uid"]]
    df["doi"] = None
    df["publication_year"] = None
    df["publication_date"] = None
    df["published_at"] = "2026-10-28T00:00:00Z"
    return df


def test_citations_some_fallback(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    df = _with_pages(make_fixture_catalog())
    df.loc[2:3, "study_id"] = "Unpublished2024"  # two datasets of one study without an entry
    df.loc[2, "doi"] = "10.1/unpub"
    with pytest.warns(
        UserWarning, match=r"2 of 3 studies have a published citation, in 2 entries .*; added 1 fallback entries"
    ):
        text, n_published, n_fallback = citation_entries(SpatialDataCohort(df), bib_url=str(bib))
    assert (n_published, n_fallback) == (2, 1)
    entries = parse_bibtex(text)
    assert list(entries) == ["Smith2023", "Wong2021", "Unpublished2024"]
    fb = entries["Unpublished2024"]
    assert fb.startswith("@misc{Unpublished2024,")
    assert f"howpublished = {{\\url{{{SITE_URL}/datasets/page-uid0003/}}}}" in fb
    assert "title = {Dataset uid0003 50\\% \\& co}" in fb
    assert "doi = {10.1/unpub}" in fb
    assert "year = {2026}" in fb
    assert "note = {SpatialData-DB datasets uid0003, uid0004; license CC0-1.0, unknown}" in fb


def test_citations_none_published(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    df = make_fixture_catalog().iloc[:2].copy()
    df["study_id"] = "NoSuchKey"
    with pytest.warns(
        UserWarning, match=r"0 of 1 studies have a published citation, in 0 entries .*; added 1 fallback entries"
    ):
        out = SpatialDataCohort(df).citations(tmp_path / "refs.bib", bib_url=str(bib))
    entries = parse_bibtex(out.read_text())
    assert list(entries) == ["NoSuchKey"]  # one study with two datasets gives one entry
    assert "title = {uid0001}" in entries["NoSuchKey"]  # no title column: the uid stands in
    assert "howpublished" not in entries["NoSuchKey"]  # no slug column: no page to link
    assert "license CC-BY-4.0, CC-BY-NC-SA-4.0}" in entries["NoSuchKey"]


def test_citations_without_study_id(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    df = _with_pages(make_fixture_catalog().iloc[:3])
    df.loc[2, "study_id"] = None
    with pytest.warns(UserWarning, match=r"1 of 2 studies"):
        text, n_published, n_fallback = citation_entries(SpatialDataCohort(df), bib_url=str(bib))
    assert (n_published, n_fallback) == (1, 1)
    entries = parse_bibtex(text)
    assert list(entries) == ["Smith2023", "uid0003"]
    assert "note = {SpatialData-DB dataset uid0003; license CC0-1.0}" in entries["uid0003"]


def _doi_bib():
    """Published entries for four GEO samples of two papers, as the published file repeats them."""
    return (
        "@article{geo:GSM3,\n  title = {Paper A},\n  DOI = {10.1/A},\n  year = {2024}\n}\n\n"
        "@article{geo:GSM1,\n  title = {Paper A},\n  doi = {https://doi.org/10.1/a},\n  note = {From {GEO}}\n}\n\n"
        "@article{geo:GSM2,\n  title = {Paper A},\n  doi = {10.1/a},\n}\n\n"
        "@article{geo:GSM4,\n  title = {Paper B},\n  doi = {10.1/b}\n}\n\n"
        "@article{geo:GSM5,\n  title = {No DOI}\n}\n"
    )


def test_published_entries_that_share_a_doi_merge(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(_doi_bib())
    df = make_fixture_catalog()
    df["study_id"] = ["geo:GSM1", "geo:GSM2", "geo:GSM3", "geo:GSM4", "geo:GSM5"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fallback entries: no warning
        text, n_published, n_fallback = citation_entries(SpatialDataCohort(df), bib_url=str(bib))
    assert (n_published, n_fallback) == (3, 0)
    entries = parse_bibtex(text)
    assert list(entries) == ["geo:GSM1", "geo:GSM4", "geo:GSM5"]  # the first key of each DOI, in key order
    assert "note = {From {GEO}; covers samples: geo:GSM1, geo:GSM2, geo:GSM3}" in entries["geo:GSM1"]
    assert "covers samples" not in entries["geo:GSM4"]
    assert "covers samples" not in entries["geo:GSM5"]


def test_merged_note_is_added_and_truncated():
    from sddb.citations import _merge_by_doi

    pairs = [(f"geo:GSM{i}", f"@article{{geo:GSM{i},\n  doi = {{10.1/x}},\n}}") for i in range(1, 9)]
    (entry,) = _merge_by_doi(pairs)
    assert entry == (
        "@article{geo:GSM1,\n  doi = {10.1/x},\n"
        "  note = {covers samples: geo:GSM1, geo:GSM2, geo:GSM3, geo:GSM4, geo:GSM5 and 3 more}\n}"
    )
    assert parse_bibtex(entry)["geo:GSM1"] == entry


def test_fallback_entries_that_share_a_doi_merge(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    df = _with_pages(make_fixture_catalog())
    df["study_id"] = ["Smith2023", "geo:GSM8", "geo:GSM9", "geo:GSM7", None]
    df["doi"] = [None, "10.1/same", "10.1/same", "10.1/other", None]
    with pytest.warns(UserWarning, match=r"1 of 5 studies .* in 1 entries .*; added 3 fallback entries") as rec:
        text, n_published, n_fallback = citation_entries(SpatialDataCohort(df), bib_url=str(bib))
    assert len(rec) == 1
    assert (n_published, n_fallback) == (1, 3)
    entries = parse_bibtex(text)
    assert list(entries) == ["Smith2023", "geo:GSM7", "geo:GSM8", "uid0005"]
    assert "uid0002; license CC-BY-NC-SA-4.0; covers samples: geo:GSM8, geo:GSM9}" in entries["geo:GSM8"]


def test_citations_text_counts_all_entries(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    df = make_fixture_catalog()
    df.loc[4, "study_id"] = "NoSuchKey"
    with pytest.warns(UserWarning, match="fallback"):
        text, n = SpatialDataCohort(df).citations_text(bib_url=str(bib))
    assert n == len(parse_bibtex(text)) == 3


def test_citations_empty_cohort_raises(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    with pytest.raises(ValueError, match="cohort is empty"):
        SpatialDataCohort(make_fixture_catalog().iloc[:0]).citations(tmp_path / "refs.bib", bib_url=str(bib))
    assert not (tmp_path / "refs.bib").exists()


def test_citations_refresh_passed(tmp_path, monkeypatch):
    import sddb.citations as cit
    from sddb.dataset import Source

    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    seen = []
    real = cit.fetch_catalog
    monkeypatch.setattr(cit, "fetch_catalog", lambda url, **kw: (seen.append(kw.get("refresh")), real(url, **kw))[1])
    SpatialDataCohort(make_fixture_catalog().iloc[:2], source=Source(refresh=True)).citations(
        tmp_path / "r.bib", bib_url=str(bib)
    )
    assert seen == [True]
