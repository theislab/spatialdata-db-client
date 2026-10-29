"""Lazy per-cohort table concatenation: reads only each member's ``tables/<name>`` group."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import fsspec
import numpy as np
import pandas as pd
from anndata.io import read_elem

from sddb.remote import _open_group, storage_options

if TYPE_CHECKING:
    import zarr

    from sddb.adapter import TableAdapter


GeneKey = Literal["ensembl", "symbol"]
Spatial = Literal["auto", "obsm", "centroids"]

_ENSEMBL = r"ENS[A-Z]*G\d{11}(\.\d+)?"
# var columns tried in order when var_names do not already hold the requested key
_ENSEMBL_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "gene_id_raw", "feature_id")
_SYMBOL_COLUMNS = ("symbol", "gene_symbols", "gene_symbol", "gene_name", "feature_name")
_MIN_OVERLAP = 0.5


def _looks_ensembl(values: Any) -> bool:
    """True when at least half of the non-null values are Ensembl gene ids."""
    s = pd.Series(values, dtype="string").dropna()
    return bool(len(s)) and bool(s.str.fullmatch(_ENSEMBL).mean() >= 0.5)


def _harmonize_var(adata: ad.AnnData, gene_key: GeneKey, uid: str) -> tuple[ad.AnnData, int, int]:
    """Re-key ``var_names`` to ``gene_key``; return the new AnnData and the (dropped, duplicate) counts.

    The old name goes to ``var["original_var_name"]``. Features without a usable key are dropped and
    repeated keys keep their first occurrence.
    """
    names = pd.Series(adata.var_names, index=adata.var_names, dtype="string")
    if gene_key == "ensembl":
        candidates = [names] + [adata.var[c] for c in _ENSEMBL_COLUMNS if c in adata.var.columns]
        key = next((c for c in candidates if _looks_ensembl(c)), None)
    else:
        candidates = [names] + [adata.var[c] for c in _SYMBOL_COLUMNS if c in adata.var.columns]
        key = next(
            (c for c in candidates if not _looks_ensembl(c) and c.astype("string").str.strip().ne("").any()),
            None,
        )
    if key is None:
        raise ValueError(
            f"member {uid!r}: no usable source for gene_key={gene_key!r}; var columns: {list(adata.var.columns)}"
        )
    key = pd.Series(key.to_numpy(), index=adata.var_names, dtype="string").str.strip()
    if gene_key == "ensembl":
        key = key.str.replace(r"\.\d+$", "", regex=True)
        usable = key.str.fullmatch(_ENSEMBL)
    else:
        usable = key.ne("")
    usable = usable.fillna(False).astype(bool)
    dup = usable & key.duplicated()
    keep = (usable & ~dup).to_numpy()
    out = adata[:, keep].copy()
    out.var["original_var_name"] = out.var_names.astype(str)
    out.var_names = key[keep].astype(str).to_numpy()
    return out, int((~usable).sum()), int(dup.sum())


def _warn_low_overlap(kept: list[tuple[str, ad.AnnData]], gene_key: GeneKey | None) -> None:
    sizes = [p.n_vars for _, p in kept]
    if len(kept) < 2 or min(sizes) == 0:
        return
    shared = len(set.intersection(*(set(p.var_names) for _, p in kept)))
    overlap = shared / min(sizes)
    if overlap < _MIN_OVERLAP:
        hint = "; pass gene_key='ensembl' or 'symbol' to align gene ids across sources" if gene_key is None else ""
        warnings.warn(
            f"members share {shared} genes, {overlap:.0%} of the smallest member's {min(sizes)}{hint}",
            UserWarning,
            stacklevel=3,
        )


def _table_name(row: Any) -> str:
    dt = getattr(row, "default_table", None)
    return str(dt) if isinstance(dt, str) and dt else "table"


def _read_table(url: str, elements: Sequence[str] | None) -> ad.AnnData:
    group = _open_group(url)
    if elements is None:
        return ad.read_zarr(group)
    return ad.AnnData(**{k: read_elem(group[k]) for k in elements if k in group})


def _region_centroids(zarr_url: str, root: zarr.Group, region: str) -> pd.DataFrame | str:
    """``x``/``y`` centroid per instance of a shapes element, in its intrinsic coordinates, or why there is none."""
    if f"shapes/{region}" not in root:
        kind = "a labels element" if f"labels/{region}" in root else "not in the store"
        return f"region {region!r} is {kind}"
    fs, path = fsspec.core.url_to_fs(zarr_url, **storage_options(zarr_url))
    path = f"{path.rstrip('/')}/shapes/{region}/shapes.parquet"
    if not fs.exists(path):  # the pre-parquet (v0.1) shapes layout
        return f"shapes {region!r} has no shapes.parquet"
    import geopandas

    cents = geopandas.read_parquet(path, filesystem=fs, columns=["geometry"]).geometry.centroid
    return pd.DataFrame({"x": cents.x.to_numpy(), "y": cents.y.to_numpy()}, index=cents.index.astype(str))


def _obs_column(adata: ad.AnnData, table: zarr.Group, key: str) -> np.ndarray:
    if key in adata.obs.columns:
        return adata.obs[key].astype(str).to_numpy()
    return pd.Series(read_elem(table[f"obs/{key}"])).astype(str).to_numpy()


def _fill_spatial(adata: ad.AnnData, zarr_url: str, table_url: str, spatial: Spatial) -> str | None:
    """Set ``obsm["spatial"]`` per ``spatial``; return why a member has no coordinates, else ``None``."""
    table = _open_group(table_url)
    stored = adata.obsm.get("spatial")
    if stored is None and "obsm/spatial" in table:
        stored = read_elem(table["obsm/spatial"])
    if stored is not None:
        stored = np.asarray(stored, dtype=np.float64)
        if spatial == "auto" and not np.isnan(stored).any(axis=1).any():
            adata.obsm["spatial"] = stored
            return None
    attrs = read_elem(table["uns/spatialdata_attrs"]) if "uns/spatialdata_attrs" in table else {}
    region, region_key, instance_key = (attrs.get(k) for k in ("region", "region_key", "instance_key"))
    reason = None
    if not region or not instance_key:
        reason = "the table annotates no region"
    else:
        regions = [region] if isinstance(region, str) else list(region)
        root = _open_group(zarr_url)
        ids = _obs_column(adata, table, instance_key)
        row_region = _obs_column(adata, table, region_key) if len(regions) > 1 else np.full(len(ids), regions[0])
        xy = np.full((adata.n_obs, 2), np.nan)
        reasons = []
        for name in regions:
            cents = _region_centroids(zarr_url, root, name)
            if isinstance(cents, str):
                reasons.append(cents)
                continue
            rows = row_region == name
            pos = cents.index.get_indexer(ids[rows])
            hit = np.flatnonzero(rows)[pos >= 0]
            xy[hit] = cents.to_numpy()[pos[pos >= 0]]
        if not np.isnan(xy).all():
            adata.obsm["spatial"] = xy
            n_miss = int(np.isnan(xy).any(axis=1).sum())
            return f"{n_miss} of {adata.n_obs} rows have no matching {instance_key!r}" if n_miss else None
        reason = "; ".join(reasons) or f"no {instance_key!r} value matches the region"
    if stored is None:
        return f"no obsm['spatial'] and {reason}"
    adata.obsm["spatial"] = stored  # no centroids: keep what is stored
    return f"{reason}; kept the stored obsm['spatial']" + (", which has NaN rows" if spatial == "auto" else "")


def concat_tables(
    members: pd.DataFrame,
    *,
    table: str | None = None,
    obs_columns: Sequence[str] = (),
    elements: Sequence[str] | None = None,
    adapter: TableAdapter | None = None,
    gene_key: GeneKey | None = None,
    spatial: Spatial = "auto",
    max_workers: int = 8,
) -> ad.AnnData:
    """Concatenate one table per member into one AnnData (outer join).

    ``obs['uid']`` labels the member; ``obs['study_id']`` is added from the members frame (all-NA if
    the column is absent). Only ``<zarr_url>/tables/<name>`` is opened, plus the table's shapes region when
    ``spatial`` needs centroids. Images are never read.

    Parameters
    ----------
    members
        One row per member with at least ``uid`` and ``zarr_url``.
    table
        Table name to read for every member; defaults to each member's ``default_table`` (else ``"table"``).
    obs_columns
        Catalog columns (e.g. ``"organism"``, ``"tissue"``) copied from ``members`` into ``obs`` as categoricals.
    elements
        AnnData elements to read, e.g. ``("X", "obs", "var")``; ``None`` takes the adapter's
        ``requirements()["elements"]``, else reads the whole table. Skipping ``layers``/``raw`` avoids
        downloading duplicate count matrices.
    adapter
        A :class:`~sddb.TableAdapter` that checks and maps each member's table as it is read, before
        concatenation, which keeps memory bounded. Skipped members and their reasons are recorded in
        ``uns["sddb"]`` together with the adapter's name and version.
    gene_key
        Re-key every member's ``var_names`` before the join, so the same gene lands in one column even when
        sources disagree (HEST keys genes by Ensembl id, 10x by symbol). ``"ensembl"`` uses the var_names
        if they are Ensembl ids, else the first Ensembl column of ``gene_ids``, ``gene_id``, ``ensembl_id``,
        ``gene_id_raw``, ``feature_id``, and strips version suffixes. ``"symbol"`` uses the var_names if
        they are not Ensembl ids, else the first of ``symbol``, ``gene_symbols``, ``gene_symbol``,
        ``gene_name``, ``feature_name``. Features without a key (e.g. control probes for ``"ensembl"``)
        are dropped and repeated keys keep the first, with one warning summarizing both. The old name is
        kept in ``var["original_var_name"]``, and ``var`` holds the first member's annotation of each
        gene. ``"var"`` is added to ``elements`` if missing. ``None`` leaves var_names unchanged, except
        that a member whose var_names repeat gets them made unique (the second copy of ``A`` becomes
        ``A-1``), with one warning naming those members.
    spatial
        Where ``obsm["spatial"]`` comes from. ``"obsm"`` keeps each table's ``obsm`` as stored, so a member
        without ``obsm["spatial"]`` gets missing (NaN) coordinates in the joined AnnData. ``"centroids"`` sets
        it to the centroids of the shapes element the table annotates (``region`` in
        ``uns["spatialdata_attrs"]``), in that element's intrinsic coordinates, which is the space a stored
        ``obsm["spatial"]`` uses. Rows are matched to shapes by the table's ``instance_key``, never by position.
        Points (e.g. cell circles) give their centre and polygons their centroid. Only that element's
        ``shapes.parquet`` is read. A labels region, or shapes in the pre-parquet layout, counts as no source,
        because its centroids would need the whole raster or a full store read. ``"auto"`` (default) keeps a
        stored ``obsm["spatial"]`` that has no NaN rows and uses centroids otherwise. With ``"auto"`` or
        ``"centroids"``, ``obsm["spatial"]`` is set even when ``elements`` leaves out ``"obsm"``: only that
        one array is read, not the rest of ``obsm``.
    max_workers
        Number of members read concurrently.

    Warns
    -----
    UserWarning
        If two or more members share fewer than half the genes of the smallest one, which usually means
        their var_names hold different id types; also when ``gene_key`` drops or dedupes features, and
        when ``gene_key=None`` makes a member's repeated var_names unique, and once when ``spatial`` finds
        no coordinate source for some members or rows (the message names the members and the reason). Their
        coordinates stay missing.

    Raises
    ------
    KeyError
        If a member's table is absent or unreadable (the message names the member uid), or an
        ``obs_columns`` entry is not a column of ``members``.
    ValueError
        If the adapter drops every member, ``gene_key`` or ``spatial`` is not a known value, or a member has no source for
        ``gene_key`` (the message names the member uid and its var columns).
    """
    if missing := [c for c in obs_columns if c not in members.columns]:
        raise KeyError(f"obs_columns not in members: {missing}")
    if gene_key not in (None, "ensembl", "symbol"):
        raise ValueError(f"gene_key must be 'ensembl', 'symbol' or None, not {gene_key!r}")
    if spatial not in ("auto", "obsm", "centroids"):
        raise ValueError(f"spatial must be 'auto', 'obsm' or 'centroids', not {spatial!r}")
    if elements is None and adapter is not None:
        elements = adapter.requirements().get("elements")
    if gene_key is not None and elements is not None and "var" not in elements:
        elements = (*elements, "var")
    rows = [row for _, row in members.iterrows()]
    dropped: dict[str, str] = {}
    rekeyed: dict[str, tuple[int, int]] = {}  # uid -> (dropped, duplicate) features under gene_key
    repeated: dict[str, int] = {}  # uid -> repeated var_names made unique (gene_key=None)
    no_xy: dict[str, str] = {}  # uid -> why some or all rows have no coordinates

    def read(row: pd.Series) -> ad.AnnData | None:
        uid = str(row["uid"])
        if adapter is not None and (issues := adapter.validate_meta(row)):
            dropped[uid] = "; ".join(issues)
            return None
        name = table or _table_name(row)
        url = f"{str(row['zarr_url']).rstrip('/')}/tables/{name}"
        try:
            adata = _read_table(url, elements)
        except Exception as e:
            raise KeyError(f"member {row['uid']!r}: table {name!r} not readable at {url}: {e}") from e
        if spatial != "obsm" and (why := _fill_spatial(adata, str(row["zarr_url"]), url, spatial)):
            no_xy[uid] = why
        if adapter is not None:
            if issues := adapter.validate(adata, row):
                dropped[uid] = "; ".join(issues)
                return None
            adata = adapter.build(adata, row)
            if adata is None:
                dropped[uid] = f"dropped by {adapter.name}.build"
                return None
        if gene_key is not None:
            adata, n_drop, n_dup = _harmonize_var(adata, gene_key, uid)
            rekeyed[uid] = (n_drop, n_dup)
        elif n_rep := int(adata.var_names.duplicated().sum()):
            repeated[uid] = n_rep  # ad.concat(join="outer") cannot reindex a non-unique var index
            adata.var_names_make_unique()
        return adata

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        parts = list(pool.map(read, rows))
    kept = [(str(row["uid"]), part) for row, part in zip(rows, parts, strict=True) if part is not None]
    if not kept:
        raise ValueError(f"every member was dropped: {dropped}")
    n_drop, n_dup = (sum(c[i] for c in rekeyed.values()) for i in (0, 1))
    if n_drop or n_dup:
        what = "an Ensembl id" if gene_key == "ensembl" else "a symbol"
        warnings.warn(
            f"gene_key={gene_key!r}: dropped {n_drop} features without {what} and {n_dup} duplicates "
            f"across {sum(1 for c in rekeyed.values() if any(c))} members",
            UserWarning,
            stacklevel=2,
        )
    if repeated:
        warnings.warn(
            f"made {sum(repeated.values())} repeated var_names unique with a '-1' style suffix in members "
            f"{sorted(repeated)}; pass gene_key='ensembl' or 'symbol' to merge them instead",
            UserWarning,
            stacklevel=2,
        )
    if no_xy := {u: w for u, w in no_xy.items() if u not in dropped}:
        warnings.warn(
            f"spatial={spatial!r}: no usable coordinates for {len(no_xy)} members: "
            + "; ".join(f"{u!r} ({w})" for u, w in sorted(no_xy.items())),
            UserWarning,
            stacklevel=2,
        )
    _warn_low_overlap(kept, gene_key)
    out = ad.concat(
        [p for _, p in kept],
        join="outer",
        label="uid",
        keys=[k for k, _ in kept],
        index_unique="-",
    )
    if gene_key is not None:  # each gene's annotation from the first member that has it
        var = pd.concat([p.var for _, p in kept])
        out.var = var[~var.index.duplicated()].reindex(out.var_names)
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
    if adapter is not None:
        out.uns["sddb"] = {"adapter": adapter.name, "adapter_version": adapter.version, "dropped": dropped}
    return out
