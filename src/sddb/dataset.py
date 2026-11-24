"""Dataset records resolved from the catalog."""

from __future__ import annotations

import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from sddb import remote
from sddb.roles import resolve as _resolve_roles

if TYPE_CHECKING:
    from spatialdata import SpatialData

# The public buckets' https origins, as the website builds them (src/lib/inventory-core.mjs httpsUrl).
_S3_HTTPS = {"scverse-spatial-eu-central-1": "https://scverse-spatial-eu-central-1.s3.eu-central-1.amazonaws.com"}
_MERFISHEYES = "https://www.merfisheyes.com/spatialdata-viewer/from-s3?url="


@dataclass(frozen=True)
class Source:
    """Where a SpatialDataCohort came from: the catalog url, cache dir, version and file hash (for sidecars/manifests).

    ``sha256`` is ``"sha256:<hex>"`` of the catalog parquet the cohort was built from, or None if unknown.
    """

    url: str | None = None
    cache_dir: str | Path | None = None
    version: str | None = None
    refresh: bool = False
    sha256: str | None = None
    release: str | None = None  # the dated release the catalog resolved to (also for catalog-latest), if known


_ATTRS = (
    "uid",
    "zarr_url",
    "technology",
    "assay",
    "organism",
    "tissue",
    "disease",
    "validation_status",
    "tier",
    "size_bytes",
)


def _clean(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return value  # container cell: present value, never scalar-isna
    return None if value is pd.NA or pd.isna(value) else value


class Dataset:
    """One catalog row: a published dataset. Column values are exposed as attributes."""

    def __init__(self, row: pd.Series, source: Source | None = None) -> None:
        self._row = row
        self._source = source  # the catalog it came from, recorded in a dest manifest

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._row.index:
            return _clean(self._row[name])
        if name in _ATTRS:
            return None
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def to_sdata(self, *, elements: list[str] | None = None, dest: str | Path | None = None) -> SpatialData:
        """Open this dataset as SpatialData, fetched and cached like :meth:`SpatialDataCohort.to_list`.

        A remote store is copied once into the cache (keyed by full uid) and reused; a local store opens in
        place. ``dest`` keeps the store in ``dest/<uid>.zarr`` (the ``download()`` layout); ``elements``
        (``"kind/name"``) copies only those elements. Both together copy only those elements into
        ``dest/<uid>.zarr`` and record them in ``dest/manifest.json`` as a ``"partial"`` entry; a later
        ``download(dest)`` fetches the whole store. If ``dest`` already holds the complete store, ``elements``
        opens it whole.
        """
        from sddb.cohort import SpatialDataCohort

        cohort = SpatialDataCohort(self._row.to_frame().T, source=self._source)
        return cohort.to_list(elements=elements, dest=dest, workers=1)[0]

    def _local_store(self, elements: Sequence[str] | None = None) -> Path:
        """The local path of the store, fetched into the cache like :meth:`to_sdata` if remote, not read.

        `elements` (``"kind/name"``) fetches only those, plus the elements their tables annotate; ``None`` the whole
        store.
        """
        from sddb.cohort import SpatialDataCohort

        cohort = SpatialDataCohort(self._row.to_frame().T, source=self._source)
        return cohort._local_stores(None if elements is None else list(elements), None, 1)[0]

    def elements(self) -> dict[str, dict[str, Any]]:
        """Return element shapes/dtypes without loading data."""
        return remote.elements(self.zarr_url)

    def resolve(self, roles: Mapping[str, str]) -> dict[str, str]:
        """Resolve role regexes to this store's element paths (``"kind/name"``); see :func:`sddb.resolve`."""
        return _resolve_roles(self.elements().keys(), roles)

    def viewer_url(self) -> str | None:
        """The dataset in MERFISHEYES, the website's viewer, which reads the store straight from S3.

        The link is the one the dataset page's "Open in MERFISHEYES" button opens. ``None`` when the store is not in
        a public bucket.
        """
        bucket, _, key = str(self.zarr_url or "").removeprefix("s3://").partition("/")
        origin = _S3_HTTPS.get(bucket) if str(self.zarr_url or "").startswith("s3://") else None
        return f"{_MERFISHEYES}{quote(f'{origin}/{key}', safe='')}" if origin else None

    def view(self) -> None:
        """Open :meth:`viewer_url` in the default browser.

        Raises
        ------
        ValueError
            If the store is not in a public bucket, so there is no viewer link.
        """
        url = self.viewer_url()
        if url is None:
            raise ValueError(f"dataset {self.uid} has no viewer link")
        webbrowser.open(url)

    def __repr__(self) -> str:
        return f"<Dataset {self.uid} technology={self.technology} organism={self.organism}>"
