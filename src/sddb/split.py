"""Deterministic, group-safe train/val/test splitting of a cohort."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_FOLDS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitManifest:
    """Per-uid fold assignment, pinned to the parent cohort hash, ``by`` key, ``seed`` and ratios."""

    parent_hash: str
    by: str
    seed: int
    ratios: dict[str, float]
    mode: str
    assignments: dict[str, str]

    def write(self, path: str | Path) -> Path:
        """Write the manifest as JSON; return the path."""
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: str | Path) -> SplitManifest:
        """Load a manifest written by :meth:`write`."""
        return cls(**json.loads(Path(path).read_text()))


def _group_key(uid: Any, study: Any) -> str:
    """Group label; a missing study becomes its own singleton group ``object:<uid>``."""
    return f"object:{uid}" if (study is None or pd.isna(study)) else str(study)


def make_split(
    members: pd.DataFrame,
    parent_hash: str,
    *,
    by: str = "study_id",
    train: float,
    val: float,
    test: float,
    seed: int,
    mode: str = "group",
) -> SplitManifest:
    """Assign whole ``by`` groups to folds (``mode="group"``) or just label them (``mode="annotate"``).

    Deterministic: group keys are sorted before a local ``default_rng(seed)`` shuffle, so the result
    is independent of row order, pandas/Python version and global RNG state.
    """
    ratios = {"train": train, "val": val, "test": test}
    groups = {str(r.uid): _group_key(r.uid, getattr(r, by, None)) for r in members.itertuples(index=False)}
    if mode == "annotate":
        return SplitManifest(parent_hash, by, seed, ratios, mode, groups)
    keys = sorted(set(groups.values()))  # deterministic input order
    nonzero = [f for f, w in ratios.items() if w > 0]
    if len(keys) < len(nonzero):
        raise ValueError(
            f"cohort has {len(keys)} group(s) but {len(nonzero)} non-zero folds requested "
            f"({nonzero}); too few to fill every fold without an empty one"
        )
    rng = np.random.default_rng(seed)
    order = [str(k) for k in rng.permutation(keys)]
    counts = _largest_remainder(len(keys), ratios)  # guarantees >=1 per non-zero fold
    fold_by_key: dict[str, str] = {}
    i = 0
    for fold in _FOLDS:
        for _ in range(counts[fold]):
            fold_by_key[order[i]] = fold
            i += 1
    return SplitManifest(parent_hash, by, seed, ratios, mode, {u: fold_by_key[g] for u, g in groups.items()})


def _largest_remainder(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Fold sizes summing to ``n``; every fold with a non-zero ratio gets at least one."""
    base = {f: int(ratios[f] * n) for f in _FOLDS}
    for f, w in ratios.items():
        if w > 0 and base[f] == 0:
            base[f] = 1
    while sum(base.values()) > n:  # trim overflow from the largest fold
        base[max(_FOLDS, key=lambda k: base[k])] -= 1
    while sum(base.values()) < n:  # give remainder to the largest ratio
        base[max(_FOLDS, key=lambda k: ratios[k])] += 1
    return base
