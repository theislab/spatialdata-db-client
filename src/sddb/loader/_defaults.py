"""What :func:`sddb.loader.materialize` takes when it is not given: the role and roles from the catalog, the mpp."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: Element roles by technology and image role, for stores written by spatialdata-io's reader of that technology. The
#: catalog has no per-member role field, so these are the reader's fixed element names: Atera ``he_image`` and
#: ``cell_labels`` (``spatialdata_io/readers/atera.py``, ``_get_aligned_images`` and ``labels["cell_labels"]``) and
#: the table ``table``; the six Atera stores of release 2026-11-12 hold exactly these (their ``sdata_repr``).
ROLES: dict[tuple[str, str], dict[str, str]] = {
    ("10x_atera", "he"): {"image": "^images/he_image$", "table": "^tables/table$", "labels": "^labels/cell_labels$"},
}
#: Relative tolerance between the catalog ``mpp`` and the one the store records. The two agree to 1e-15 on the
#: Atera stores (``scripts/atera_loader_cohort.py``), so anything above this is a different value, not rounding.
MPP_RTOL = 1e-6
#: Visium v1/v2 arrays: neighbouring spots of a row are two ``array_col`` apart and "spaced 100 µm apart" (10x
#: glossary), so one ``array_col`` step is 50 µm. The same fit as spatialdata-io's visium reader (``_px_per_um``).
VISIUM_ARRAY_COL_UM = 50.0
#: Largest median distance, in µm, of a spot from the affine fit of its pixel centre on its array position. A real
#: array fits to a small fraction of a pixel; 5 µm is a tenth of an ``array_col`` step.
PITCH_MAX_RESIDUAL_UM = 5.0
#: Relative tolerance between a catalog ``mpp`` and the pitch fit. Above it the catalog value is a different one: the
#: stale catalog values are 9 to 27% off the fit, while full-resolution OME pixel sizes agree with it within 0.6%.
PITCH_RTOL = 0.02
#: The catalog ``mpp`` column that is known stale, by technology: the last catalog release that has it. Up to
#: release 2026-11-12 the Visium ``mpp`` is 55 µm / ``spot_diameter_fullres``, a display size of the spot rather
#: than the 55 µm spot, so it is 1.18 or 1.27 times too small against the array pitch. For these the catalog value is
#: never used, and a mismatch with the pitch fit is logged rather than raised. A catalog without a release counts as
#: stale too.
STALE_CATALOG_MPP: dict[str, str] = {"10x_visium": "2026-11-12"}


def default_role(members: Sequence[Any]) -> Literal["he", "if"]:
    """``"he"`` if every member has an H&E image (catalog ``hne_image``), else ``"if"`` if every one has an IF image."""
    for role, column in (("he", "hne_image"), ("if", "if_image")):
        if members and all(_flag(getattr(m, column, None)) for m in members):
            return role  # type: ignore[return-value]
    raise ValueError(
        "pass role: the members do not all have an H&E image (catalog hne_image) or all an IF image (if_image)"
    )


def _flag(value: Any) -> bool:
    """A catalog boolean (numpy, pandas or Python; ``None`` when null) as a Python bool."""
    return value is not None and bool(value)


def default_roles(members: Sequence[Any], role: str) -> dict[str, str]:
    """The roles of the members' technology for `role` (:data:`ROLES`); they must share one."""
    found = {(str(getattr(m, "technology", None)), role) for m in members}
    if len(found) != 1:
        raise ValueError(f"pass roles: the members have several technologies, {sorted(t for t, _ in found)}")
    (key,) = found
    if key not in ROLES:
        raise ValueError(
            f"pass roles: no default roles for technology {key[0]!r} with role {role!r}; known {sorted(ROLES)}"
        )
    return dict(ROLES[key])


def catalog_mpp(member: Any) -> float | None:
    """The member's catalog ``mpp``, ``None`` if null."""
    mpp = getattr(member, "mpp", None)
    return None if mpp is None or (isinstance(mpp, float) and math.isnan(mpp)) else float(mpp)


def store_mpp(sdata: Any, image: str) -> float | None:
    """The image's microns per scale-0 pixel that the store itself records, ``None`` if it records none.

    spatialdata-io's Atera reader stores the microns per ``global`` unit as ``sdata.attrs["pixel_size"]``
    (``spatialdata_io/readers/atera.py``: ``sdata.attrs[AteraKeys.PIXEL_SIZE] = pixel_size``). The image's mpp is that
    times its ``global`` units per pixel, ``sqrt(|det A|)`` of the 2x2 part of its affine onto ``global``, the scale
    :class:`~sddb.loader.Rasterize` uses. Other readers do not record it.
    """
    import numpy as np
    from spatialdata.transformations import get_transformation

    pixel_size = sdata.attrs.get("pixel_size")
    if pixel_size is None:
        return None
    transforms = get_transformation(sdata.images[image], get_all=True)
    if "global" not in transforms:
        return None
    a = np.asarray(transforms["global"].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
    return float(pixel_size) * math.sqrt(abs(float(np.linalg.det(a[:2, :2]))))


def catalog_is_stale(technology: Any, release: str | None) -> bool:
    """Whether the catalog ``mpp`` of `technology` in `release` is known stale (:data:`STALE_CATALOG_MPP`).

    Releases are compared as ISO dates; a release that is not one, or none, counts as stale.
    """
    last = STALE_CATALOG_MPP.get(str(technology))
    if last is None:
        return False
    return release is None or not _ISO_DATE.match(str(release)) or str(release)[:10] <= last


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def pitch_mpp(sdata: Any, image: str) -> float | None:
    """The image's microns per scale-0 pixel from the Visium array pitch of the store's own spots.

    For each table with ``obs["array_col"]`` and ``obs["array_row"]``, the spots are the shapes the table annotates
    (their centres, matched by the instance key). A least-squares affine fit of the centres on ``(array_col,
    array_row)`` gives the length of one ``array_col`` step, which is :data:`VISIUM_ARRAY_COL_UM`; the image's scale
    relative to the spots in a coordinate system both have gives its mpp. This is the fit of spatialdata-io's visium
    reader (``_px_per_um``); the spot circles' radius and ``spot_diameter_fullres`` are not used.

    Returns ``None`` when no table has the array columns, or none of their spots shares a coordinate system with the
    image.

    Raises
    ------
    ValueError
        If the array positions are not a Visium v1/v2 array (a spot's ``array_col`` and ``array_row`` have the same
        parity), or the centres fit their array positions worse than :data:`PITCH_MAX_RESIDUAL_UM`.
    """
    import numpy as np
    from spatialdata.models import get_table_keys
    from spatialdata.transformations import get_transformation

    image_tf = get_transformation(sdata.images[image], get_all=True)
    for name, table in sdata.tables.items():
        if not {"array_col", "array_row"} <= set(table.obs.columns):
            continue
        region, region_key, instance_key = get_table_keys(table)
        for r in [region] if isinstance(region, str) else list(region):
            if r not in sdata.shapes:
                continue
            spots = sdata.shapes[r]
            spots_tf = get_transformation(spots, get_all=True)
            shared = sorted(set(image_tf) & set(spots_tf))
            if not shared:
                continue
            obs = table.obs[table.obs[region_key] == r] if region_key in table.obs else table.obs
            col_row = obs[["array_col", "array_row"]].to_numpy(dtype=np.float64)
            centres = spots.geometry.loc[obs[instance_key].to_numpy()].centroid
            xy = np.column_stack([centres.x.to_numpy(), centres.y.to_numpy()])
            units_per_col, residual = _array_fit(col_row, xy, f"{name}/{r}")
            um_per_unit = VISIUM_ARRAY_COL_UM / units_per_col
            if residual * um_per_unit > PITCH_MAX_RESIDUAL_UM:
                raise ValueError(
                    f"the spots of {r!r} fit their array positions to a median {residual * um_per_unit:.3g} µm, more "
                    f"than {PITCH_MAX_RESIDUAL_UM:g} µm, so their pitch does not give the pixel size; pass mpp_native"
                )
            cs = "global" if "global" in shared else shared[0]
            image_scale, spots_scale = (_scale(tf[cs]) for tf in (image_tf, spots_tf))
            return um_per_unit * image_scale / spots_scale
    return None


def _array_fit(col_row: Any, xy: Any, where: str) -> tuple[float, float]:
    """``(length of one array_col step, median residual)`` of the affine fit of `xy` on `col_row`, in `xy`'s units."""
    import numpy as np

    col, row = col_row[:, 0].astype(int), col_row[:, 1].astype(int)
    if len(col) < 3 or np.any((col - row) % 2):
        raise ValueError(
            f"{where}: the spots are not on a Visium v1/v2 array (array_col and array_row of a spot must be both even "
            "or both odd, and at least 3 spots), so the pixel size cannot be derived from the array pitch"
        )
    design = np.column_stack([col_row, np.ones(len(col_row))])
    coef, *_ = np.linalg.lstsq(design, xy, rcond=None)
    step = float(np.linalg.norm(coef[0]))
    if not step > 0:
        raise ValueError(f"{where}: the spots all share one array column, so the array pitch cannot be fitted")
    residual = float(np.median(np.linalg.norm(design @ coef - xy, axis=1)))
    return step, residual


def _scale(tf: Any) -> float:
    """``sqrt(|det A|)`` of the 2x2 part of `tf`'s affine: the target units per source unit."""
    import numpy as np

    a = np.asarray(tf.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
    return math.sqrt(abs(float(np.linalg.det(a[:2, :2]))))


def resolve_mpp(
    member: Any, sdata: Any, image: str, *, given: Mapping[str, float | None], release: str | None
) -> tuple[float | None, str | None]:
    """The member's microns per scale-0 image pixel and where it came from, as the contract's ``mpp_source``.

    In order:

    1. ``"given"``: `given` has the uid (``None`` there means unknown). Not checked.
    2. ``"store_pitch_fit"``: a ``10x_visium`` member is fitted on its own spots (:func:`pitch_mpp`). The catalog
       ``mpp`` must agree to :data:`PITCH_RTOL`, unless it is known stale for `release` (:func:`catalog_is_stale`),
       when the difference is only logged. A Visium store without array positions has no mpp: its stale catalog
       value is not used.
    3. ``"catalog_checked_store"``: the catalog ``mpp``, which agrees with the pixel size the store records
       (:func:`store_mpp`) to :data:`MPP_RTOL`, else it raises.
    4. ``"catalog_unchecked"``: the catalog ``mpp`` of a store that records none.

    ``(None, None)`` when the mpp is unknown.
    """
    uid = str(member.uid)
    if uid in given:
        value = given[uid]
        return (None, None) if value is None else (float(value), "given")
    catalog = catalog_mpp(member)
    technology = getattr(member, "technology", None)
    stale = catalog_is_stale(technology, release)
    if str(technology) == "10x_visium":
        fit = pitch_mpp(sdata, image)
        if fit is None:
            if stale or catalog is None:
                logger.warning(
                    f"{uid}: no array positions to fit the pixel size on, and the catalog mpp of release {release} "
                    "is known stale; the mpp is unknown (native grid): pass mpp_native={uid: ...}"
                    if stale
                    else f"{uid}: no array positions to fit the pixel size on and no catalog mpp; the mpp is unknown"
                )
                return None, None
            logger.info(f"{uid}: mpp_native {catalog} from the catalog; the store has no array positions to check it")
            return catalog, "catalog_unchecked"
        if catalog is not None and not math.isclose(fit, catalog, rel_tol=PITCH_RTOL):
            message = (
                f"{uid}: the catalog's mpp {catalog} differs from the pitch fit of the store's spots, {fit} "
                f"(x {fit / catalog:.4f})"
            )
            if not stale:
                raise ValueError(f"{message}, by more than {PITCH_RTOL:g}; pass mpp_native={{{uid!r}: ...}}")
            logger.warning(f"{message}; the catalog value of release {release} is known stale, the fit is used")
        return fit, "store_pitch_fit"
    if catalog is None:
        return None, None
    return catalog, check_mpp(uid, catalog, sdata, image, given=False)


def check_mpp(uid: str, mpp: float | None, sdata: Any, image: str, *, given: bool) -> str | None:
    """Where `mpp` came from, as the contract's ``mpp_source`` records it; raise if the catalog's disagrees with the store.

    Returns ``"given"`` for a value the caller passed (an override: not checked), ``"catalog_checked_store"`` for a
    catalog value that agrees with the one the store records (:func:`store_mpp`) to :data:`MPP_RTOL`,
    ``"catalog_unchecked"`` for a catalog value of a store that records none, and ``None`` when the mpp is unknown.
    """
    if mpp is None:
        return None
    if given:
        return "given"
    own = store_mpp(sdata, image)
    if own is None:
        logger.info(f"{uid}: mpp_native {mpp} from the catalog; the store records no pixel size to check it")
        return "catalog_unchecked"
    if not math.isclose(own, mpp, rel_tol=MPP_RTOL):
        raise ValueError(
            f"{uid}: the catalog's mpp {mpp} differs from the store's own {own} (pixel_size x the image's scale "
            f"onto global) by more than {MPP_RTOL:g}; pass mpp_native={{{uid!r}: ...}} with the right value"
        )
    return "catalog_checked_store"
