"""Lazy per-cohort table concatenation: reads only each member's ``tables/<name>`` group."""

from __future__ import annotations

from typing import Any

import anndata as ad
import pandas as pd

from sddb.remote import _open_group


def _table_name(row: Any) -> str:
    dt = getattr(row, "default_table", None)
    return str(dt) if isinstance(dt, str) and dt else "table"


def concat_tables(members: pd.DataFrame, *, table: str | None = None, verify: bool = True) -> ad.AnnData:
    """Concatenate one table per member into a single AnnData (outer join, ``obs["uid"]`` labels the member).

    Only ``<zarr_url>/tables/<name>`` is opened, so images/shapes are never touched. Works for local
    paths and (anonymous) remote URLs.

    Parameters
    ----------
    members
        DataFrame with ``uid`` and ``zarr_url`` (optionally ``default_table``, ``fingerprint``).
    table
        Table name to read from every member; defaults to each member's ``default_table`` else ``"table"``.
    verify
        If True, run :func:`sddb.freeze.verify_members` first (raises ``CohortDriftError`` on drift).

    Raises
    ------
    KeyError
        If a member's table is absent or unreadable; the message names the member uid.
    """
    if verify:
        from sddb.freeze import verify_members

        verify_members(members)
    parts: list[ad.AnnData] = []
    keys: list[str] = []
    for row in members.itertuples(index=False):
        name = table or _table_name(row)
        url = f"{str(row.zarr_url).rstrip('/')}/tables/{name}"
        try:
            parts.append(ad.read_zarr(_open_group(url)))
        except Exception as e:
            raise KeyError(f"member {row.uid!r}: table {name!r} not readable at {url}: {e}") from e
        keys.append(str(row.uid))
    return ad.concat(parts, join="outer", label="uid", keys=keys, index_unique="-")
