"""Lazy per-cohort table concatenation: reads only each member's ``tables/<name>`` group."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import anndata as ad
import pandas as pd
from anndata.io import read_elem

from sddb.remote import _open_group


def _table_name(row: Any) -> str:
    dt = getattr(row, "default_table", None)
    return str(dt) if isinstance(dt, str) and dt else "table"


def _read_table(url: str, elements: Sequence[str] | None) -> ad.AnnData:
    group = _open_group(url)
    if elements is None:
        return ad.read_zarr(group)
    return ad.AnnData(**{k: read_elem(group[k]) for k in elements if k in group})


def concat_tables(
    members: pd.DataFrame,
    *,
    table: str | None = None,
    obs_columns: Sequence[str] = (),
    elements: Sequence[str] | None = None,
    transform: Callable[[ad.AnnData, pd.Series], ad.AnnData | None] | None = None,
    max_workers: int = 8,
) -> ad.AnnData:
    """Concatenate one table per member into one AnnData (outer join).

    ``obs['uid']`` labels the member; ``obs['study_id']`` is added from the members frame (all-NA if
    the column is absent). Only ``<zarr_url>/tables/<name>`` is opened — images/shapes are untouched.

    Parameters
    ----------
    members
        One row per member with at least ``uid`` and ``zarr_url``.
    table
        Table name to read for every member; defaults to each member's ``default_table`` (else ``"table"``).
    obs_columns
        Catalog columns (e.g. ``"organism"``, ``"tissue"``) copied from ``members`` into ``obs`` as categoricals.
    elements
        AnnData elements to read, e.g. ``("X", "obs", "var")``; ``None`` reads the whole table. Skipping
        ``layers``/``raw`` avoids downloading duplicate count matrices.
    transform
        Called as ``transform(adata, row)`` on each member right after reading and before concatenation,
        e.g. QC or gene subsetting to bound memory. Return ``None`` to drop the member.
    max_workers
        Number of members read concurrently.

    Raises
    ------
    KeyError
        If a member's table is absent or unreadable (the message names the member uid), or an
        ``obs_columns`` entry is not a column of ``members``.
    """
    if missing := [c for c in obs_columns if c not in members.columns]:
        raise KeyError(f"obs_columns not in members: {missing}")
    rows = [row for _, row in members.iterrows()]

    def read(row: pd.Series) -> ad.AnnData | None:
        name = table or _table_name(row)
        url = f"{str(row['zarr_url']).rstrip('/')}/tables/{name}"
        try:
            adata = _read_table(url, elements)
        except Exception as e:
            raise KeyError(f"member {row['uid']!r}: table {name!r} not readable at {url}: {e}") from e
        return transform(adata, row) if transform else adata

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        parts = list(pool.map(read, rows))
    kept = [(str(row["uid"]), part) for row, part in zip(rows, parts, strict=True) if part is not None]
    out = ad.concat([p for _, p in kept], join="outer", label="uid", keys=[k for k, _ in kept], index_unique="-")
    study_by_uid = (
        dict(zip(members["uid"].astype("string"), members["study_id"].astype("string"), strict=True))
        if "study_id" in members.columns
        else {}
    )
    out.obs["study_id"] = out.obs["uid"].astype("string").map(study_by_uid).astype("string")
    for col in obs_columns:
        if col != "study_id":
            out.obs[col] = (
                out.obs["uid"]
                .astype("string")
                .map(dict(zip(members["uid"].astype("string"), members[col], strict=True)))
                .astype("category")
            )
    return out
