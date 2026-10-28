"""Task-adapter interfaces: the contracts a downstream task repo implements.

The client defines only these Protocols. Building/materializing a task view from a cohort lives
downstream of the cohort, not in the client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sddb.objectmeta import ObjectMeta

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData

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


@runtime_checkable
class TableAdapter(Protocol):
    """Per-table counterpart of :class:`~sddb.TaskAdapter`, used by :meth:`~sddb.SpatialDataCohort.to_anndata`.

    Each member's table is checked and mapped into a shared space (genes, QC, ...) as it is read, before
    concatenation. Any issue from ``validate_meta`` or ``validate`` skips the member; the reasons land in
    ``uns["sddb"]["dropped"]`` of the result.
    """

    name: str
    version: str

    def requirements(self) -> dict[str, Any]:
        """Declare what to read, e.g. ``{"elements": ("X", "obs", "var")}``; ``{}`` reads the whole table."""
        ...

    def validate_meta(self, row: pd.Series) -> list[Issue]:
        """Cheap check on the member's catalog row, before its table is read."""
        ...

    def validate(self, adata: AnnData, row: pd.Series) -> list[Issue]:
        """Authoritative check on the table as read."""
        ...

    def build(self, adata: AnnData, row: pd.Series) -> AnnData | None:
        """Map the table into the task's space; ``None`` drops the member (e.g. nothing left after QC)."""
        ...
