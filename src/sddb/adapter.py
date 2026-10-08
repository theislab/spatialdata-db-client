"""Task-adapter interface: turn a frozen cohort into a task-specific view with provenance."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

import sddb
from sddb import remote
from sddb.concat import _table_name
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


_ELEMENT_KINDS = remote._ELEMENT_KINDS


def _metas(members: pd.DataFrame, **kw: Any) -> list[ObjectMeta]:
    return fetch_many([(str(r.uid), str(r.zarr_url)) for r in members.itertuples(index=False)], **kw)


def _match(name: str, select: str, how: str) -> bool:
    """exact/suffix/substring are case-insensitive; regex uses the pattern's own flags."""
    n, s = name.lower(), select.lower()
    if how == "exact":
        return n == s
    if how == "suffix":  # anchored: `square_016um` must not catch `square_016um_for_vitessce`
        return n == s or (n.endswith(s) and not n[len(n) - len(s) - 1].isalnum())
    if how == "substring":
        return s in n
    if how == "regex":
        return re.search(select, name) is not None
    raise ValueError(f"unknown match mode {how!r}")


def _sdata_elements(sdata: Any) -> list[str]:
    """Sorted ``"kind/name"`` of every element present in an opened SpatialData."""
    return sorted(f"{kind}/{name}" for kind in _ELEMENT_KINDS for name in getattr(sdata, kind))


def _resolve_elements(url: str, roles: list[dict[str, Any]]) -> list[str]:
    from sddb.remote import elements as _elements

    inv = _elements(url)  # {"kind/name": {...}}
    resolved: list[str] = []
    for role in roles:
        kind = role["role"]
        kind = kind if kind in _ELEMENT_KINDS else kind + "s"  # image -> images, shape -> shapes, ...
        select, how = role["select"], role.get("match", "suffix")
        hits = [p for p in inv if p.split("/", 1)[0] == kind and _match(p.split("/", 1)[1], select, how)]
        if len(hits) == 1:
            resolved.append(hits[0])
        elif not hits:
            raise ValueError(f"role {role!r}: no element matched; present: {sorted(inv)}")
        else:
            raise ValueError(f"role {role!r}: matched {len(hits)} elements {hits}; tighten `select`/`match`")
    return resolved


def _open(url: str, adapter_obj: TaskAdapter) -> Any:
    from sddb.remote import open_sdata

    req = getattr(adapter_obj, "requirements", None)
    roles = ((req() if callable(req) else {}) or {}).get("elements")
    if not roles:
        return open_sdata(url, lazy=False)
    return open_sdata(url, lazy=False, elements=_resolve_elements(url, roles))


def check(members: pd.DataFrame, adapter: TaskAdapter, *, deep: bool = False) -> PreflightReport:
    """Compatibility report. ``deep=False`` uses metadata only (advisory); ``deep=True`` opens each object."""
    excluded: dict[Issue, list[str]] = {}
    compatible = 0
    if deep:
        pairs = [(str(r.uid), adapter.validate(_open(str(r.zarr_url), adapter))) for r in members.itertuples(index=False)]
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
    """Task view over ``members``. Lazy by default; ``materialize=True`` builds all and writes provenance.

    With ``materialize=True`` each ``adapter.build(sdata)`` runs once for its side effects and provenance; the
    returned ``TaskView`` stays lazy (re-invoking a thunk re-opens the store).
    """
    if split is not None and split.parent_hash != cohort_hash:
        raise ValueError(f"split.parent_hash {split.parent_hash!r} does not match cohort hash {cohort_hash!r}")
    if verify:
        from sddb.freeze import verify_members

        verify_members(members)

    def thunk(url: str) -> Callable[[], Any]:
        return lambda: adapter.build(_open(url, adapter))

    view = TaskView([(str(r.uid), thunk(str(r.zarr_url))) for r in members.itertuples(index=False)])
    if not materialize:
        return view
    out_dir = Path(out or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    reqs = adapter.requirements() or {}
    roles = reqs.get("elements")
    produced = []
    for r in members.itertuples(index=False):
        sdata = _open(str(r.zarr_url), adapter)
        out_item = adapter.build(sdata)
        produced.append(
            {
                "uid": str(r.uid),
                "table": _table_name(r),
                "output": type(out_item).__name__,
                "elements": _sdata_elements(sdata) if roles else None,
            }
        )
    split_id = (
        None
        if split is None
        else {"by": split.by, "seed": split.seed, "ratios": split.ratios, "mode": split.mode}
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "cohort_hash": cohort_hash,
                "created_at": datetime.now(UTC).isoformat(),
                "split": split_id,
                "adapter": {"name": adapter.name, "version": adapter.version, "config": adapter.config},
                "sddb_version": sddb.__version__,
                "requirements": reqs,
                "members": produced,
            },
            indent=2,
        )
    )
    return view
