"""Task-adapter interface: the contract a downstream task repo implements.

The client defines only this Protocol. Building/materializing a task view from a cohort lives
downstream of the cohort, not in the client.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sddb.objectmeta import ObjectMeta

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
