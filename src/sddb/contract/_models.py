"""Pydantic models of the metadata a contract dataset carries in its table's ``uns``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, StrictInt, model_validator

CONTRACT_VERSION = "0.1.0"


class AdapterRecord(BaseModel):
    """One adapter step that produced a dataset, with the parameters it ran with."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    params: dict[str, Any]


class DatasetContract(BaseModel):
    """``uns["sddb_contract"]`` of a dataset: the contract version, its source and how it was written.

    Attributes
    ----------
    version
        The contract version, semver. On 0.x a breaking change bumps the minor version.
    uid, study_id, catalog_release
        The source catalog member and the catalog release it was read from.
    role
        The image role of the collection: ``"he"`` (uint8, 3 channels) or ``"if"`` (uint16, one per `channels`).
    grid
        ``"target"`` if the image was resampled to `mpp_target`, ``"native"`` if it keeps its own grid (mpp unknown).
    mpp_target, mpp_native
        Microns per pixel of the target grid and of the source image, ``None`` if unknown. The target grid needs
        both, so that upsampling can be checked.
    upsampled, allow_upsample
        Whether the target grid is finer than the native one, and whether the user allowed that.
    allow_mixed_grids
        Whether this dataset may share a collection with datasets on the other kind of grid.
    dtype_rule
        How the source pixels were converted to the role's dtype, ``None`` if they already had it.
    channels
        IF channel names, in image order.
    adapters
        The adapter steps, in the order they ran.
    stain_reference
        The H&E stain normalisation reference (stain matrix, white point, max concentrations), if one was applied.
    n_centroid_fallback
        How many cells have no pixel in ``labels/cells`` (smaller than a pixel of the written grid), so that
        :class:`~sddb.loader.Centroids` took their ``obsm["spatial"]`` from the source: the centre of mass in the
        source labels at native resolution, the polygon centroid or the source table's centroid. Optional: 0 when the
        key is absent.
    n_outside_image
        How many cells have ``obs["outside_image"]`` True: their centroid is not on the H&E or IF image's footprint
        (beyond the written grid, or on the fill of a rotated image's corners). Their ``obsm["spatial"]`` is still
        their true position, so it may lie outside ``[0, w) x [0, h)``. Optional: 0 when the key is absent.
    n_features_present, n_features_filled
        How many features of the collection's var this dataset's source measured (``var["present"]`` True), and how
        many it did not, which are 0 in ``X``. Optional: 0 when the key is absent.
    """

    model_config = ConfigDict(extra="forbid")

    version: str
    uid: str
    study_id: str
    catalog_release: str
    role: Literal["he", "if"]
    grid: Literal["target", "native"]
    mpp_target: PositiveFloat | None = None
    mpp_native: PositiveFloat | None = None
    upsampled: bool
    allow_upsample: bool
    allow_mixed_grids: bool
    dtype_rule: str | None = None
    channels: list[str] | None = None
    adapters: list[AdapterRecord]
    stain_reference: dict[str, list[float] | list[list[float]] | str] | None = None
    n_centroid_fallback: StrictInt = Field(default=0, ge=0)
    n_outside_image: StrictInt = Field(default=0, ge=0)
    n_features_present: StrictInt = Field(default=0, ge=0)
    n_features_filled: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.grid == "target" and self.mpp_target is None:
            raise ValueError("grid 'target' requires mpp_target")
        if self.grid == "target" and self.mpp_native is None:
            raise ValueError("grid 'target' requires mpp_native; a dataset whose mpp is unknown keeps its native grid")
        if self.upsampled and not self.allow_upsample:
            raise ValueError("upsampled requires allow_upsample")
        if self.role == "if" and not self.channels:
            raise ValueError("role 'if' requires channels")
        return self

    def to_uns(self) -> dict[str, Any]:
        """The model as anndata can store it in ``uns``: each adapter record is a JSON string.

        anndata writes a list of dicts as the strings of their reprs, so the records are encoded explicitly.
        """
        out = self.model_dump()
        out["adapters"] = [json.dumps(a, sort_keys=True) for a in out["adapters"]]
        return out

    @classmethod
    def from_uns(cls, uns: Mapping[str, Any]) -> Self:
        """Parse ``uns["sddb_contract"]`` as read by anndata; the inverse of :meth:`to_uns`."""
        data = {k: _plain(v) for k, v in uns.items()}
        if isinstance(data.get("adapters"), list):
            data["adapters"] = [json.loads(a) if isinstance(a, str) else a for a in data["adapters"]]
        return cls.model_validate(data)


class Scalefactors(BaseModel):
    """``uns["spatial"][uid]["scalefactors"]``: the resolution of the written grid."""

    model_config = ConfigDict(extra="forbid")

    mpp: PositiveFloat | None
    mpp_native: PositiveFloat | None
    native_to_written: PositiveFloat


class SpatialMetadata(BaseModel):
    """``uns["spatial"][uid]["metadata"]``: the names of the dataset's elements."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["he", "if"]
    image: str
    labels: Literal["cells"]
    tissue: str | None = None


class SpatialEntry(BaseModel):
    """``uns["spatial"][uid]``, Squidpy's layout without the ``images`` arrays (the image is ``images/<role>``)."""

    model_config = ConfigDict(extra="forbid")

    scalefactors: Scalefactors
    metadata: SpatialMetadata

    @classmethod
    def from_uns(cls, uns: Mapping[str, Any]) -> Self:
        """Parse ``uns["spatial"][uid]`` as read by anndata."""
        return cls.model_validate(_plain(uns))


def _plain(value: Any) -> Any:
    """Turn what anndata reads back (numpy arrays and scalars, nested dicts) into plain Python values."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    return value
