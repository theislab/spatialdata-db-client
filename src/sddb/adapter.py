"""Task-adapter interface: turn a frozen cohort into a task-specific view with provenance."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

import sddb
from sddb.objectmeta import ObjectMeta, fetch_many

Issue = str


@runtime_checkable
class TaskAdapter(Protocol):
    """What a task repo implements. ``validate_meta`` is cheap/advisory, ``validate`` authoritative."""

    name: str
    version: str
    config: dict[str, Any]

    def requirements(self) -> dict[str, Any]:
        """Declare what the task needs (tables, elements, ...)."""
        ...

    def validate_meta(self, meta: ObjectMeta) -> list[Issue]:
        """Cheap, metadata-only check (advisory)."""
        ...

    def validate(self, sdata: Any) -> list[Issue]:
        """Authoritative check on the opened object."""
        ...

    def build(self, sdata: Any) -> Any:
        """Produce the task-specific item for one object."""
        ...


@dataclass
class PreflightReport:
    """Counts of compatible members and excluded uids grouped by reason."""

    total: int
    compatible: int
    excluded: dict[Issue, list[str]]  # reason -> uids
    tier: str  # "advisory" | "authoritative"

    def __repr__(self) -> str:
        lines = [f"{self.total} candidate / {self.compatible} compatible  [{self.tier}]"]
        lines += [f"  {len(uids)} {reason}: {', '.join(uids)}" for reason, uids in sorted(self.excluded.items())]
        return "\n".join(lines)


def _metas(members: pd.DataFrame, **kw: Any) -> list[ObjectMeta]:
    return fetch_many([(str(r.uid), str(r.zarr_url)) for r in members.itertuples(index=False)], **kw)


def _open(url: str) -> Any:
    from sddb.remote import open_sdata

    return open_sdata(url, lazy=False)


def check(members: pd.DataFrame, adapter: TaskAdapter, *, deep: bool = False) -> PreflightReport:
    """Compatibility report. ``deep=False`` uses metadata only (advisory); ``deep=True`` opens each object."""
    excluded: dict[Issue, list[str]] = {}
    compatible = 0
    if deep:
        pairs = [(str(r.uid), adapter.validate(_open(str(r.zarr_url)))) for r in members.itertuples(index=False)]
        tier = "authoritative"
    else:
        pairs = [(m.uid, adapter.validate_meta(m)) for m in _metas(members)]
        tier = "advisory"
    for uid, issues in pairs:
        if issues:
            for i in issues:
                excluded.setdefault(i, []).append(uid)
        else:
            compatible += 1
    return PreflightReport(len(members), compatible, excluded, tier)


class TaskView:
    """Lazy iterable of ``(uid, thunk)``; calling a thunk opens the object and runs ``adapter.build``."""

    def __init__(self, items: list[tuple[str, Callable[[], Any]]]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[tuple[str, Callable[[], Any]]]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> tuple[str, Callable[[], Any]]:
        return self._items[i]


def build(
    members: pd.DataFrame,
    adapter: TaskAdapter,
    *,
    split: Any = None,
    materialize: bool = False,
    out: str | Path | None = None,
    verify: bool = True,
    cohort_hash: str = "",
) -> TaskView:
    """Task view over ``members``. Lazy by default; ``materialize=True`` builds all and writes provenance."""
    if split is not None and split.parent_hash != cohort_hash:
        raise ValueError(f"split.parent_hash {split.parent_hash!r} does not match cohort hash {cohort_hash!r}")
    if verify:
        from sddb.freeze import verify_members

        verify_members(members)

    def thunk(url: str) -> Callable[[], Any]:
        return lambda: adapter.build(_open(url))

    view = TaskView([(str(r.uid), thunk(str(r.zarr_url))) for r in members.itertuples(index=False)])
    if not materialize:
        return view
    out_dir = Path(out or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = [{"uid": uid, "output": type(fn()).__name__} for uid, fn in view]
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "cohort_hash": cohort_hash,
                "split_hash": split.parent_hash if split is not None else None,
                "adapter": {"name": adapter.name, "version": adapter.version, "config": adapter.config},
                "sddb_version": sddb.__version__,
                "members": produced,
            },
            indent=2,
        )
    )
    return view
