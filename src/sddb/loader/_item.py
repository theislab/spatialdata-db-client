"""The item the loader adapters pass along: one cohort member's table, image and labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from sddb.roles import resolve

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

    from sddb.contract import AdapterRecord

_ROLES = {"image": "images", "table": "tables", "labels": "labels"}
_MAX_LABEL = 2**32 - 1  # labels/cells is written as uint32
# obs columns that spatialdata-io fills with each row's value in the cell labels when the table annotates shapes:
# atera's "label_id" (readers/atera.py) and xenium's "cell_labels" (readers/xenium.py:473, cells_as_circles=True)
_LABEL_COLUMNS = ("label_id", "cell_labels")


@dataclass(frozen=True)
class _Source:
    """What :class:`~sddb.loader.Rasterize` reads from the source SpatialData.

    The affines are 3x3 in ``(x, y)`` and map an element's intrinsic coordinates (scale 0) onto the coordinate
    system `cs` shared by the image and the cells.
    """

    cs: str
    levels: list[xr.DataArray]  # the image's multiscale levels (c, y, x), finest first
    image_to_cs: np.ndarray
    cells_kind: Literal["labels", "shapes"]
    cells: Any  # the labels' scale 0 (y, x) DataArray, or the shapes' GeoDataFrame
    cells_to_cs: np.ndarray
    instance_key: str
    # the source table's obsm["spatial"] in row order (None without one), and the affine of the element the table
    # annotates (its region), whose intrinsic coordinates those are taken to be
    table_centroids: np.ndarray | None = None
    region_to_cs: np.ndarray | None = None
    region: str | None = None


@dataclass
class LoaderItem:
    """One cohort member on its way through the loader adapters.

    Each adapter returns a new item and appends its :class:`~sddb.contract.AdapterRecord` to `records`. Task code
    writes the last item as one dataset of a loader collection.

    Attributes
    ----------
    uid, study_id
        The catalog member.
    role
        ``"he"`` or ``"if"``.
    table
        The cells, one row per cell of the annotated element. ``obs["label_id"]`` holds each row's value in
        ``labels["cells"]`` (see :meth:`from_sdata`); :class:`~sddb.loader.Centroids` adds ``obsm["spatial"]``.
    image
        ``(c, y, x)``, dask-backed. Before :class:`~sddb.loader.Rasterize` it is the source image's scale 0, after it
        the image on the written grid.
    labels
        ``"cells"`` after :class:`~sddb.loader.Rasterize`, ``"tissue"`` after :class:`~sddb.loader.TissueMask`, both
        ``(y, x)`` on the image's grid.
    mpp_native
        Microns per scale-0 pixel of the source image, ``None`` if unknown.
    mpp_target
        Microns per pixel of the written grid on the ``"target"`` grid, ``None`` on the native grid.
    grid
        ``"target"`` or ``"native"``, set by :class:`~sddb.loader.Rasterize`.
    records
        The adapter steps, in the order they ran.
    flags
        ``allow_upsample``, ``allow_mixed_grids`` and ``upsampled``.
    channels
        IF channel names in image order (from the image's ``c`` coordinate), ``None`` for H&E.
    dtype_rule
        How the source pixels were converted to the role's dtype, ``None`` if they already had it.
    stain_reference
        The stain reference applied by :class:`~sddb.loader.StainNormalize`, as plain lists.
    shape_centroids
        ``(n_obs, 2)`` ``(x, y)`` centroid of each row's polygon in written pixel-index coordinates, set by
        :class:`~sddb.loader.Rasterize` when the cells are shapes (NaN for a row without a shape), ``None``
        otherwise. :class:`~sddb.loader.Centroids` uses it for cells that have no pixel on the written grid.
    table_centroids
        ``(n_obs, 2)`` the source table's ``obsm["spatial"]`` mapped into written pixel-index coordinates by
        :class:`~sddb.loader.Rasterize`, as intrinsic coordinates of the element the table annotates; ``None``
        without one. `table_centroids_element` names that element, for :class:`~sddb.loader.Centroids`' check that
        the mapping agrees with the cells that have pixels.
    native_cells, native_cells_to_index
        When the cells are a labels element: its scale 0 ``(y, x)``, and the 3x3 ``(x, y)`` affine from its intrinsic
        coordinates to written pixel-index coordinates (its own transformation), set by :class:`~sddb.loader.Rasterize`.
        :class:`~sddb.loader.Centroids` takes a cell's centre of mass there when the cell has no pixel on the written
        grid. ``None`` for shape cells.
    n_centroid_fallback
        How many rows took their centroid from `native_cells`, `shape_centroids` or `table_centroids`, set by
        :class:`~sddb.loader.Centroids`.
    image_footprint
        ``(4, 2)`` ``(x, y)`` corners of the source image's scale-0 extent in written pixel-index coordinates, set by
        :class:`~sddb.loader.Rasterize`. The written grid is the bounding box of this quadrilateral, so with a rotated
        image its corners are fill. :class:`~sddb.loader.Centroids` flags cells whose centroid is not on it.
    n_outside_image
        How many rows have ``obs["outside_image"]`` True, set by :class:`~sddb.loader.Centroids`.
    image_source
        The source image's multiscale levels and the affine :class:`~sddb.loader.Rasterize` resampled them with, set by
        it, so that :meth:`StainNormalize.fit <sddb.loader.StainNormalize.fit>` can read a coarse level instead of the
        written image.
    tissue_overview
        The tissue mask on :class:`~sddb.loader.TissueMask`'s overview and the overview's factor ``f`` (an overview
        pixel covers ``f x f`` written pixels), set by it.
    label_rule
        Where ``obs["label_id"]`` came from (see :meth:`from_sdata`).
    """

    uid: str
    study_id: str
    role: Literal["he", "if"]
    table: AnnData
    image: xr.DataArray
    labels: dict[str, xr.DataArray]
    mpp_native: float | None
    mpp_target: float | None = None
    grid: Literal["target", "native"] | None = None
    records: list[AdapterRecord] = field(default_factory=list)
    flags: dict[str, bool] = field(
        default_factory=lambda: {"allow_upsample": False, "allow_mixed_grids": False, "upsampled": False}
    )
    channels: list[str] | None = None
    dtype_rule: str | None = None
    stain_reference: dict[str, Any] | None = None
    shape_centroids: np.ndarray | None = field(default=None, repr=False)
    table_centroids: np.ndarray | None = field(default=None, repr=False)
    table_centroids_element: str | None = None
    native_cells: xr.DataArray | None = field(default=None, repr=False)
    native_cells_to_index: np.ndarray | None = field(default=None, repr=False)
    n_centroid_fallback: int = 0
    image_footprint: np.ndarray | None = field(default=None, repr=False)
    n_outside_image: int = 0
    label_rule: str = ""
    image_source: Any = field(default=None, repr=False)
    tissue_overview: tuple[np.ndarray, int] | None = field(default=None, repr=False)
    source: _Source | None = field(default=None, repr=False)

    @classmethod
    def from_sdata(
        cls,
        sdata: SpatialData,
        *,
        uid: str,
        study_id: str,
        role: Literal["he", "if"],
        roles: Mapping[str, str],
        mpp_native: float | None,
        label_column: str | None = None,
    ) -> LoaderItem:
        """Resolve the image, the table and the cells of one member.

        Parameters
        ----------
        sdata
            The member, opened lazily.
        uid, study_id, role
            See the attributes.
        roles
            Regexes (see :func:`sddb.resolve`) for ``"image"`` and ``"table"``, and optionally ``"labels"``: an
            existing cell raster to resample instead of rasterising the shapes the table annotates.
        mpp_native
            Microns per scale-0 pixel of the image, ``None`` if unknown (the item then keeps its native grid).
        label_column
            The ``obs`` column holding each row's value in the cell raster; by default it is inferred, see below.

        Notes
        -----
        ``obs["label_id"]``, the value of each row's cell in ``labels["cells"]``, is taken from the first that applies:

        1. `label_column`;
        2. the instance ids, if the cells are the labels element the table annotates;
        3. ``obs["label_id"]``, else ``obs["cell_labels"]``: the spatialdata-io conventions when the labels values
           differ from the instance ids (Atera: the table annotates ``cell_boundaries`` by ``cell_id``,
           ``cell_labels`` holds ``label_id``; Xenium read with ``cells_as_circles=True``: the table annotates
           ``cell_circles`` by ``cell_id`` and ``obs["cell_labels"]`` holds each row's value in ``cell_labels``).
           With existing labels it is checked on a sample: the labels at 200 rows' shape centroids, or, without
           shapes, the values of a central 512 x 512 window, must match for at least 95%;
        4. with existing labels nothing else applies, so a ``ValueError`` asks for `label_column`;
        5. the instance ids, if they are positive integers that fit uint32 (shapes are rasterised with them);
        6. the row position plus one (shapes are rasterised with it), e.g. for Xenium's string cell ids.

        The ids must be unique, positive and fit uint32. The table keeps ``X``, ``obs`` and ``var``; its ``obsm``
        (often in microns) and ``uns`` are dropped, as the written dataset gets its own. ``obsm["spatial"]`` is kept
        on the source, for :class:`~sddb.loader.Centroids`' fallback for cells without pixels.

        Raises
        ------
        ValueError
            If a role is unknown or names the wrong kind of element, the table does not annotate exactly one
            labels or shapes element, the image and the cells share no coordinate system, or the label ids cannot
            be inferred or are invalid.
        """
        import anndata as ad
        from spatialdata.models import get_table_keys
        from spatialdata.transformations import get_transformation

        unknown = set(roles) - set(_ROLES)
        if unknown or not {"image", "table"} <= set(roles):
            raise ValueError(f"roles must have 'image' and 'table' and may have 'labels'; got {sorted(roles)}")
        paths = resolve([f"{kind}/{name}" for kind, name, _ in sdata.gen_elements()], roles)
        for r, path in paths.items():
            if not path.startswith(_ROLES[r] + "/"):
                raise ValueError(f"role {r!r} must name an element of {_ROLES[r]!r}; got {path!r}")
        names = {r: p.split("/", 1)[1] for r, p in paths.items()}

        source_table = sdata.tables[names["table"]]
        region, _, instance_key = get_table_keys(source_table)
        regions = [region] if isinstance(region, str) else list(region)
        if len(regions) != 1 or not (regions[0] in sdata.labels or regions[0] in sdata.shapes):
            raise ValueError(f"table {names['table']!r} must annotate one labels or shapes element; got {regions}")
        region = regions[0]

        cells_kind: Literal["labels", "shapes"]
        if "labels" in names:
            cells_kind, cells_name = "labels", names["labels"]
        else:
            cells_kind, cells_name = ("labels" if region in sdata.labels else "shapes"), region
        cells_el = sdata.labels[cells_name] if cells_kind == "labels" else sdata.shapes[cells_name]
        image_el = sdata.images[names["image"]]

        shared = set(get_transformation(image_el, get_all=True)) & set(get_transformation(cells_el, get_all=True))
        if not shared:
            raise ValueError(f"image {names['image']!r} and {cells_name!r} share no coordinate system")
        cs = "global" if "global" in shared else sorted(shared)[0]

        obs = source_table.obs
        instance = obs[instance_key]
        if label_column is not None:
            ids, rule = obs[label_column], f"obs[{label_column!r}]"
        elif cells_kind == "labels" and cells_name == region:
            ids, rule = instance, f"instance id obs[{instance_key!r}] (the table annotates the labels)"
        elif column := next((c for c in _LABEL_COLUMNS if c in obs), None):
            ids, rule = obs[column], f"obs[{column!r}]"
        elif cells_kind == "labels":
            raise ValueError(
                f"cannot tell which value of labels {cells_name!r} each row of {names['table']!r} has: the table "
                f"annotates {region!r} and has none of {[f'obs[{c!r}]' for c in _LABEL_COLUMNS]}; pass label_column"
            )
        elif pd.api.types.is_integer_dtype(instance) and len(instance) and instance.min() > 0:
            ids, rule = instance, f"instance id obs[{instance_key!r}]"
            if instance.max() > _MAX_LABEL:
                ids, rule = pd.Series(np.arange(1, len(obs) + 1), index=obs.index), "row position + 1"
        else:
            ids, rule = pd.Series(np.arange(1, len(obs) + 1), index=obs.index), "row position + 1"
        ids = _check_ids(ids, rule)

        new_obs = obs.copy()
        new_obs["label_id"] = ids
        table = ad.AnnData(X=source_table.X, obs=new_obs, var=source_table.var.copy())

        table_centroids = None
        if "spatial" in source_table.obsm:
            table_centroids = np.asarray(source_table.obsm["spatial"], dtype=np.float64)[:, :2]
        annotated_el = sdata.labels[region] if region in sdata.labels else sdata.shapes[region]
        levels = _levels(image_el)
        channels = [str(c) for c in levels[0].coords["c"].values] if role == "if" else None
        cells = _levels(cells_el)[0] if cells_kind == "labels" else cells_el
        source = _Source(
            cs=cs,
            levels=levels,
            image_to_cs=_affine(image_el, cs),
            cells_kind=cells_kind,
            cells=cells,
            cells_to_cs=_affine(cells_el, cs),
            instance_key=instance_key,
            table_centroids=table_centroids,
            region_to_cs=_affine(annotated_el, cs),
            region=region,
        )
        if rule in {f"obs[{c!r}]" for c in _LABEL_COLUMNS} and cells_kind == "labels":
            region_el = sdata.shapes[region] if region in sdata.shapes else None
            region_to_cs = _affine(region_el, cs) if region_el is not None else None
            _check_label_column(source, instance, ids, region_el, region_to_cs, cells_name)
        return cls(
            uid=uid,
            study_id=study_id,
            role=role,
            table=table,
            image=levels[0],
            labels={},
            mpp_native=mpp_native,
            channels=channels,
            label_rule=rule,
            source=source,
        )


_CHECK_ROWS = 200  # rows sampled to check that obs["label_id"] holds the labels' values
_CHECK_SHARE = 0.95  # share that must match; a concave cell's centroid can lie outside its own label
_CHECK_WINDOW = 512  # side of the labels window read when no shapes locate the rows


def _check_label_column(
    source: _Source, instance: pd.Series, ids: np.ndarray, shapes: Any, shapes_to_cs: np.ndarray | None, name: str
) -> None:
    """Check on a sample that the label column holds each row's value in the labels, without a full read.

    The column is ``obs["label_id"]`` or ``obs["cell_labels"]`` (see :meth:`LoaderItem.from_sdata`).

    With shapes, the labels are read at the centroids of a seeded sample of rows (point reads). Without, a window
    at the centre of the labels is read and its label values must be among the ids.

    Raises
    ------
    ValueError
        If fewer than 95% match.
    """
    import dask.array as da

    labels = source.cells
    data = labels.data if isinstance(labels.data, da.Array) else da.from_array(np.asarray(labels.data))
    h, w = data.shape
    if shapes is not None and shapes_to_cs is not None:
        if not shapes.index.is_unique:
            dup = shapes.index[shapes.index.duplicated()].unique()
            raise ValueError(
                f"the shapes the table annotates have {len(dup)} duplicated index values (e.g. {list(dup[:3])}); "
                "each row must locate one shape"
            )
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(len(ids), min(_CHECK_ROWS, len(ids)), replace=False))
        found = shapes.index.get_indexer(instance.to_numpy()[rows])
        rows, found = rows[found >= 0], found[found >= 0]
        if not len(rows):
            return
        cent = shapes.geometry.iloc[found].centroid
        xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy(), np.ones(len(rows))])
        pix = (np.linalg.inv(source.cells_to_cs) @ shapes_to_cs @ xy.T)[:2].T  # labels intrinsic (x, y)
        px = np.clip(np.floor(pix[:, 0]).astype(np.int64), 0, w - 1)
        py = np.clip(np.floor(pix[:, 1]).astype(np.int64), 0, h - 1)
        values = np.asarray(data.vindex[py, px].compute())
        share, what = float(np.mean(values == ids[rows])), f"{len(rows)} sampled rows' centroids"
    else:
        y0, x0 = max(0, h // 2 - _CHECK_WINDOW // 2), max(0, w // 2 - _CHECK_WINDOW // 2)
        window = np.asarray(data[y0 : y0 + _CHECK_WINDOW, x0 : x0 + _CHECK_WINDOW].compute())
        values = np.unique(window[window != 0])
        if not len(values):
            return
        share, what = float(np.isin(values, ids).mean()), f"{len(values)} label values in a central window"
    if share < _CHECK_SHARE:
        raise ValueError(
            f"the label column does not hold the values of labels {name!r}: only {share:.0%} of {what} match; "
            "pass label_column"
        )


def _check_ids(ids: pd.Series, rule: str) -> np.ndarray:
    if not pd.api.types.is_integer_dtype(ids):
        raise ValueError(f"label ids from {rule} must be integers; got {ids.dtype}")
    out = ids.to_numpy(dtype=np.int64)
    if len(out) and (out.min() <= 0 or out.max() > _MAX_LABEL):
        raise ValueError(f"label ids from {rule} must be in [1, {_MAX_LABEL}]; got [{out.min()}, {out.max()}]")
    if len(np.unique(out)) != len(out):
        raise ValueError(f"label ids from {rule} are not unique")
    return out


def _levels(element: Any) -> list[xr.DataArray]:
    """The levels of a raster element, finest first; a single-scale element is one level."""
    if isinstance(element, xr.DataArray):
        return [element]
    keys = sorted(element.keys(), key=lambda k: int("".join(ch for ch in k if ch.isdigit()) or 0))
    return [next(iter(element[k].ds.data_vars.values())) for k in keys]


def _affine(element: Any, cs: str) -> np.ndarray:
    from spatialdata.transformations import get_transformation

    tf = get_transformation(element, to_coordinate_system=cs)
    return np.asarray(tf.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=np.float64)
