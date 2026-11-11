"""Marker sets (``markers.yaml``) and their coverage of a gene panel.

A compartment is covered by a panel when at least one of its marker genes is in the panel, or when
``markers.yaml`` records a fallback for that compartment and panel. Genes match without regard to case,
so human symbols match mouse panels. Control probes (negative controls, blank and unassigned
codewords) are not genes and are left out of panel ids and coverage.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import yaml


def data_path(name: str) -> Path:
    """A data file (cohorts.yaml, markers.yaml or thresholds.yaml).

    It sits next to the package in an installed wheel, or in the primers/ directory of a checkout.
    """
    here = Path(__file__).resolve().parent
    return here / name if (here / name).exists() else here.parent / name


MARKERS = data_path("markers.yaml")
_CONTROL = re.compile(
    r"^(negcontrol|neg_control|negprb|unassigned|deprecatedcodeword|intergenic|blank|systemcontrol)", re.I
)


def panel_genes(genes: Iterable[str]) -> list[str]:
    """The genes of a panel without its control probes (Xenium NegControl*, MERSCOPE Blank-*, CosMx NegPrb*)."""
    return [str(g) for g in genes if not _CONTROL.match(str(g))]


def load_markers(path: str | Path = MARKERS) -> dict[str, dict]:
    """The marker sets by name, each with ``citation``, ``compartments`` and optional ``fallback``."""
    return yaml.safe_load(Path(path).read_text())["sets"]


def panel_id(technology: str, genes: Iterable[str]) -> str:
    """A short, stable name for a gene panel, e.g. ``Xenium-377-1a2b3c4d``."""
    upper = sorted({g.upper() for g in panel_genes(genes)})
    digest = hashlib.sha1("\n".join(upper).encode()).hexdigest()[:8]
    return f"{technology}-{len(upper)}-{digest}"


def coverage(marker_set: dict, genes: Iterable[str]) -> dict[str, list[str]]:
    """Each compartment's marker genes that are in the panel, spelled as in the panel."""
    by_upper = {g.upper(): g for g in panel_genes(genes)}
    return {
        name: [by_upper[g.upper()] for g in markers if g.upper() in by_upper]
        for name, markers in marker_set["compartments"].items()
    }


def uncovered(marker_set: dict, genes: Iterable[str], pid: str) -> list[str]:
    """Compartments with no marker in the panel and no valid fallback for panel ``pid``.

    A fallback is valid when it has a ``note`` and either its ``genes`` are non-empty and all in the
    panel, or it says ``not_measured: true`` (the panel has no usable marker, and readings must say so).
    """
    genes = list(genes)
    in_panel = {g.upper() for g in panel_genes(genes)}
    fallback = (marker_set.get("fallback") or {}).get(pid) or {}

    def valid(entry: object) -> bool:
        fb = entry if isinstance(entry, dict) else {}
        picked = [str(g).upper() for g in fb.get("genes") or []]
        if not fb.get("note"):
            return False
        return fb.get("not_measured") is True or (bool(picked) and set(picked) <= in_panel)

    return [name for name, hits in coverage(marker_set, genes).items() if not hits and not valid(fallback.get(name))]
