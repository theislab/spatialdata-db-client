import json
from pathlib import Path

PRIMERS = Path(__file__).parent.parent / "docs" / "primers"
MAX_BYTES = 20 * 1024 * 1024  # executed notebook with outputs (design 2026-11-10)


def test_primers_index_is_in_the_docs():
    assert (PRIMERS / "index.md").is_file()
    assert "primers/index" in (PRIMERS.parent / "index.md").read_text()


def test_each_primer_has_its_environment_and_fits_the_size_cap():
    for d in (p for p in PRIMERS.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        for name in ("primer.ipynb", "pixi.toml", "pixi.lock"):
            assert (d / name).is_file(), f"{d.name}: missing {name}"
        assert (d / "primer.ipynb").stat().st_size <= MAX_BYTES, f"{d.name}: primer.ipynb above 20 MB"


def test_each_primer_is_committed_executed():
    for nb in PRIMERS.glob("*/primer.ipynb"):
        cells = [c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
        assert cells, f"{nb.parent.name}: primer.ipynb has no code cells"
        assert all(c.get("execution_count") for c in cells), f"{nb.parent.name}: primer.ipynb is not executed"
