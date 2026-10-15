"""Lazy per-cohort table concatenation: reads only each member's ``tables/<name>`` group."""

from __future__ import annotations

from typing import Any

import anndata as ad
import pandas as pd

from sddb.remote import _open_group


def _table_name(row: Any) -> str:
    dt = getattr(row, "default_table", None)
    return str(dt) if isinstance(dt, str) and dt else "table"


def concat_tables(members: pd.DataFrame, *, table: str | None = None) -> ad.AnnData:
    """Concatenate one table per member into one AnnData (outer join).

    ``obs['uid']`` labels the member; ``obs['study_id']`` is added from the members frame (all-NA if
    the column is absent). Only ``<zarr_url>/tables/<name>`` is opened — images/shapes are untouched.

    Raises
    ------
    KeyError
        If a member's table is absent or unreadable; the message names the member uid.
    """
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
    out = ad.concat(parts, join="outer", label="uid", keys=keys, index_unique="-")
    study_by_uid = (
        dict(zip(members["uid"].astype("string"), members["study_id"].astype("string"), strict=True))
        if "study_id" in members.columns
        else {}
    )
    out.obs["study_id"] = out.obs["uid"].astype("string").map(study_by_uid).astype("string")
    return out
