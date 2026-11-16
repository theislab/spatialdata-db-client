"""One-member prototype of the Atera loader collection: materialize in this process, validate, read a few batches.

Usage, inside one compute-node job (``SDDB_CACHE_DIR`` on the shared filesystem)::

    python scripts/atera_loader_prototype.py <dest> [--uid jCLxdY1tWKmbfWOB0001] [--version 2026-11-12] [--workers 16]

The member comes from the catalog release through the cohort (its store is copied into the sddb cache once), and is
written whole with ``default_chain(0.5)`` and ``runner="local"``. A single member's var is its own features (the
union of one member). Writes ``<dest>.prototype.json`` with the counts and the time of each step.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

ROLES = {"image": "^images/he_image$", "table": "^tables/table$", "labels": "^labels/cell_labels$"}


def main() -> None:
    """Materialize one member locally, validate it and compare it with its catalog record."""
    import zarr
    from anndata.io import read_elem

    import sddb
    from sddb.contract import validate
    from sddb.loader import default_chain

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dest", type=Path)
    p.add_argument("--uid", default="jCLxdY1tWKmbfWOB0001")
    p.add_argument("--version", default="2026-11-12")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cohort = sddb.Catalog(version=args.version).cohort(uids=[args.uid])
    (record,) = list(cohort)
    out: dict = {"uid": args.uid, "title": record.title, "catalog_n_obs": int(record.n_obs)}
    out["catalog_n_features"] = int(record.n_features)
    t0 = time.time()
    cohort.materialize(
        args.dest,
        adapters=default_chain(0.5),
        role="he",
        roles=ROLES,
        mpp_native={args.uid: float(record.mpp)},
        runner="local",
        workers=args.workers,
    )
    out["materialize_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    out["validate"] = validate(args.dest)
    out["validate_s"] = round(time.time() - t0, 1)
    g = zarr.open_group(args.dest, mode="r")["dataset_0"]
    contract = read_elem(g["uns"]["sddb_contract"])
    var = read_elem(g["var"])
    out.update(
        cells_written=int(g["obsm"]["spatial"].shape[0]),
        features_written=int(len(var)),
        present=int(var["present"].sum()),
        n_features_filled=contract.get("n_features_filled"),
        n_centroid_fallback=contract.get("n_centroid_fallback"),
        in_tissue_share=float(read_elem(g["obs"]["in_tissue"]).mean()),
        he_shape=list(g["images"]["he"].shape),
    )
    out["cells_equal_catalog"] = out["cells_written"] == out["catalog_n_obs"]
    print(json.dumps(out, indent=1, default=str), flush=True)
    Path(f"{args.dest}.prototype.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
