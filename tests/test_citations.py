from __future__ import annotations

from tests._fixtures import make_fixture_catalog, make_fixture_citations_bib

from sddb.citations import parse_bibtex
from sddb.dataset import Results


def test_parse_nested():
    entries = parse_bibtex(make_fixture_citations_bib())
    assert set(entries) == {"Smith2023", "Lee2022", "Wong2021", "Extra1999"}
    assert "{{nested}}" in entries["Lee2022"]
    assert entries["Lee2022"].endswith("}")
    assert "{The Consortium}" in entries["Smith2023"]


def test_results_citations(tmp_path):
    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    res = Results(make_fixture_catalog().iloc[:2])  # Smith2023 only
    out = res.citations(tmp_path / "refs.bib", bib_url=str(bib))
    text = out.read_text()
    assert "Smith2023" in text
    assert "Lee2022" not in text
    assert "Extra1999" not in text
    assert list(parse_bibtex(text)) == ["Smith2023"]


def test_citations_warns_on_no_match(tmp_path):
    import pytest

    bib = tmp_path / "citations.bib"
    bib.write_text(make_fixture_citations_bib())
    df = make_fixture_catalog().iloc[:2].copy()
    df["study_id"] = "NoSuchKey"
    with pytest.warns(UserWarning, match="no cite-keys match"):
        out = Results(df).citations(tmp_path / "refs.bib", bib_url=str(bib))
    assert out.read_text() == ""
