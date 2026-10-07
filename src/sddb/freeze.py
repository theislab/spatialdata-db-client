"""FrozenCohort: a citable, hash-pinned snapshot of a cohort selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sddb.dataset import Source

SCHEMA_VERSION = "1"
_SIDECAR_THRESHOLD = 200  # members beyond this go to a parquet sidecar


def cohort_hash(filter: dict[str, Any], members: pd.DataFrame) -> str:
    """sha256 over the filter and ``(uid, lamin_version, fingerprint)`` of each member, sorted by uid."""
    rows = [
        {
            "uid": str(r.uid),
            "lamin_version": (None if pd.isna(r.lamin_version) else str(r.lamin_version)),
            "fingerprint": str(r.fingerprint),
        }
        for r in members.sort_values("uid").itertuples(index=False)
    ]
    payload = json.dumps({"filter": filter, "members": rows}, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FrozenCohort:
    """Immutable record of a cohort: filter, per-member fingerprints and a deterministic ``hash``."""

    schema_version: str
    created_at: str
    source: Source
    filter: dict[str, Any]
    hash: str
    members: pd.DataFrame  # read-only by contract; frozen=True does NOT protect in-place mutation

    def write(self, path: str | Path) -> Path:
        """Write JSON to ``path`` (members go to a parquet sidecar when large); return the path."""
        p = Path(path)
        head = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source": {"url": self.source.url, "version": self.source.version},
            "filter": self.filter,
            "hash": self.hash,
            "n_members": len(self.members),
        }
        if len(self.members) > _SIDECAR_THRESHOLD:
            side = p.with_suffix(".members.parquet")
            self.members.to_parquet(side)
            head["members_ref"] = side.name
        else:
            head["members"] = json.loads(self.members.to_json(orient="records"))
        p.write_text(json.dumps(head, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> FrozenCohort:
        """Read a cohort written by :meth:`write`."""
        p = Path(path)
        head = json.loads(p.read_text())
        if "members_ref" in head:
            members = pd.read_parquet(p.parent / head["members_ref"])
        else:
            members = pd.DataFrame(head["members"])
        src = Source(url=head["source"]["url"], version=head["source"]["version"])
        return cls(head["schema_version"], head["created_at"], src, head["filter"], head["hash"], members)


class CohortDriftError(RuntimeError):
    """A frozen cohort's objects changed since freeze; `drifted` maps uid -> reason."""

    def __init__(self, drifted: dict[str, str]) -> None:
        self.drifted = drifted
        super().__init__(f"{len(drifted)} object(s) drifted since freeze: {sorted(drifted)}")


def verify_members(members: pd.DataFrame, *, workers: int = 8) -> None:
    """Re-fetch each member's metadata and raise CohortDriftError on fingerprint mismatch."""
    from sddb.objectmeta import fetch_many

    pairs = [(str(r.uid), str(r.zarr_url)) for r in members.itertuples(index=False)]
    now = {m.uid: m for m in fetch_many(pairs, workers=workers)}
    drifted: dict[str, str] = {}
    for r in members.itertuples(index=False):
        cur = now[str(r.uid)]
        if cur.fingerprint != str(r.fingerprint):
            drifted[str(r.uid)] = f"fingerprint {r.fingerprint} -> {cur.fingerprint}"
    if drifted:
        raise CohortDriftError(drifted)
