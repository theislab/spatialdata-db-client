import importlib.util
from pathlib import Path

import yaml
from conftest import IMAGING_GENES, member, symbols
from primers.markers import MARKERS, coverage, load_markers, panel_id, uncovered

ROOT = Path(__file__).resolve().parents[1]
SET = {"compartments": {"epithelial": ["EPCAM", "KRT20"], "immune": ["PTPRC", "CD3E"]}}


def _panel_check():
    """Import scripts/panel_check.py, which is not part of the package."""
    spec = importlib.util.spec_from_file_location("panel_check", ROOT / "scripts" / "panel_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_markers_cover_every_compartment_per_panel():
    """A compartment is covered by one marker in the panel or by a fallback for that exact panel."""
    full, partial = ["EPCAM", "PTPRC", "ACTB"], ["Epcam", "Actb"]  # the mouse spelling matches
    assert uncovered(SET, full, panel_id("Xenium", full)) == []
    pid = panel_id("Xenium", partial)
    assert coverage(SET, partial) == {"epithelial": ["Epcam"], "immune": []}
    assert uncovered(SET, partial, pid) == ["immune"]
    with_fallback = {**SET, "fallback": {pid: {"immune": {"genes": ["Actb"], "note": "chosen after seeing the panel"}}}}
    assert uncovered(with_fallback, partial, pid) == []
    # a fallback for another panel does not count
    assert uncovered(with_fallback, partial, panel_id("Merscope", partial)) == ["immune"]


def test_panel_id_ignores_order_and_case():
    """The panel id depends on the technology and the set of genes only."""
    assert panel_id("Xenium", ["A", "b"]) == panel_id("Xenium", ["B", "a"]) != panel_id("Merscope", ["A", "B"])


def test_every_approved_cohort_has_a_cited_marker_set():
    """Each of the 10 approved cohorts names a marker set with a DOI and non-empty compartments."""
    markers = load_markers(MARKERS)
    cohorts = yaml.safe_load((ROOT / "cohorts.yaml").read_text())["cohorts"]
    approved = [c for c in cohorts if not c.get("reserve")]
    assert len(approved) == 10
    for c in approved:
        s = markers[c["marker_set"]]
        assert s["citation"]
        assert all(d.startswith("10.") for d in s["citation"])
        assert s["compartments"]
        assert all(s["compartments"].values())


def test_panel_check_reads_each_member_panel(make_catalog):
    """Symbols and Ensembl counts per member; a member without symbols is reported, not raised."""
    cat = make_catalog(
        [
            member("a", "Xenium"),  # HEST style: Ensembl var_names, symbols in var
            member("b", "Merscope", ensembl=False),  # Allen style: symbols only
            member("c", "Xenium", keys=False),  # no symbols: reported, not raised
        ]
    )
    pc = _panel_check()
    genes = symbols(IMAGING_GENES)
    marker_sets = {"x": {"compartments": {"one": [genes[0]], "missing": ["NOPE"]}}}
    spec = {"slug": "s", "query": {}, "marker_set": "x", "families": ["imaging"]}
    members, panels = pc.check([spec], marker_sets, cat, max_workers=1)
    m = members.set_index("uid")
    assert m.loc["a", ["n_genes", "n_ensembl"]].tolist() == [len(genes), len(genes)]
    assert m.loc["b", ["n_genes", "n_ensembl"]].tolist() == [len(genes), 0]
    assert m.loc["c", "error"].startswith("ValueError")
    # a and b share one gene list but are different technologies, so two panels
    assert len(panels) == 2
    assert set(panels["uncovered"]) == {"missing"}
    assert "| cohort |" in pc.markdown_table(panels)


def test_fallback_needs_a_note_and_panel_genes():
    """An empty, note-less or off-panel fallback does not cover a compartment."""
    panel = ["EPCAM", "ACTB", "B2M"]
    pid = panel_id("Xenium", panel)
    for entry in ({}, {"genes": ["B2M"]}, {"genes": [], "note": "n"}, {"genes": ["CD3E"], "note": "n"}):
        assert uncovered({**SET, "fallback": {pid: {"immune": entry}}}, panel, pid) == ["immune"]
    assert uncovered({**SET, "fallback": {pid: {"immune": {"genes": ["b2m"], "note": "n"}}}}, panel, pid) == []


def test_control_probes_are_not_genes():
    """Control probes change neither the panel id nor coverage."""
    genes = ["EPCAM", "PTPRC"]
    with_controls = [*genes, "NegControlProbe_00042", "BLANK_0006", "Blank-12", "UnassignedCodeword_0001", "NegPrb3"]
    assert panel_id("Xenium", with_controls) == panel_id("Xenium", genes)
    assert coverage({"compartments": {"blank": ["BLANK_0006"]}}, with_controls) == {"blank": []}


def test_not_measured_needs_a_note():
    """A compartment may be declared not measurable on a panel, but only with a note."""
    panel = ["EPCAM", "ACTB"]
    pid = panel_id("Xenium", panel)
    bare = {**SET, "fallback": {pid: {"immune": {"not_measured": True}}}}
    noted = {**SET, "fallback": {pid: {"immune": {"not_measured": True, "note": "no immune genes on this panel"}}}}
    assert uncovered(bare, panel, pid) == ["immune"]
    assert uncovered(noted, panel, pid) == []


def test_data_path_prefers_the_package_then_the_checkout(tmp_path, monkeypatch):
    """data_path finds a file next to the package (installed wheel) before the checkout's primers/ copy."""
    import primers.markers as pm

    assert pm.data_path("cohorts.yaml").exists()
    pkg = tmp_path / "primers"
    pkg.mkdir()
    (tmp_path / "markers.yaml").write_text("checkout")
    monkeypatch.setattr(pm, "__file__", str(pkg / "markers.py"))
    assert pm.data_path("markers.yaml") == tmp_path / "markers.yaml"
    (pkg / "markers.yaml").write_text("wheel")
    assert pm.data_path("markers.yaml") == pkg / "markers.yaml"
