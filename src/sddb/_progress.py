"""Download progress on stderr: a tqdm byte bar when tqdm is installed, else two plain lines."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pandas as pd


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.3g} {unit}"
        n /= 1000
    return f"{n:.3g} TB"


def _sizes(df: pd.DataFrame) -> list[int]:
    """Catalog ``size_bytes`` per row, 0 where missing."""
    if "size_bytes" not in df:
        return [0] * len(df)
    # to_numeric first: a row taken from a Dataset arrives with object dtype, and fillna on it warns in pandas 2.2+
    return pd.to_numeric(df["size_bytes"], errors="coerce").fillna(0).astype(int).tolist()


@contextmanager
def _progress(n: int, total: int, desc: str = "downloading") -> Iterator[Any]:
    """Yield an object with ``.update(nbytes)`` for ``n`` stores of ``total`` bytes. Silent when ``n`` is 0."""
    if n == 0:
        yield SimpleNamespace(update=lambda _: None)
        return
    try:
        from tqdm import tqdm
    except ImportError:
        print(f"{desc}: {n} datasets" + (f", {_fmt_size(total)}" if total else ""), file=sys.stderr)
        yield SimpleNamespace(update=lambda _: None)  # ponytail: plain mode has no per-store ticks
        print(f"{desc}: done", file=sys.stderr)
        return
    with tqdm(total=total or None, unit="B", unit_scale=True, file=sys.stderr, desc=desc) as bar:
        yield bar
