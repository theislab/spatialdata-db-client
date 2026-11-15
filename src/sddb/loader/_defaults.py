"""What :func:`sddb.loader.materialize` takes from the catalog when it is not given: the role, the roles, the mpp."""

from __future__ import annotations

import logging
import math
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


def default_mpp(members: Sequence[Any], given: Mapping[str, float | None] | None) -> dict[str, float | None]:
    """Microns per scale-0 image pixel by uid: `given` where it has the uid, else the catalog ``mpp`` (``None`` if null)."""
    given = dict(given or {})
    out: dict[str, float | None] = {}
    for m in members:
        uid = str(m.uid)
        if uid in given:
            out[uid] = given[uid]
            continue
        mpp = getattr(m, "mpp", None)
        out[uid] = None if mpp is None or (isinstance(mpp, float) and math.isnan(mpp)) else float(mpp)
    return out


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
