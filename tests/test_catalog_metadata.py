"""Test Catalog metadata reading (generated_at from release stamping)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from sddb import Catalog


def _write(path, meta):
    """Write a minimal parquet with given metadata."""
    t = pa.table(
        {
            "uid": ["a"],
            "technology": ["Xenium"],
            "assay": ["FISH"],
            "organism": ["human"],
            "validation_status": ["pass"],
            "zarr_url": ["s3://example/a.zarr"],
        }
    ).replace_schema_metadata(meta)
    pq.write_table(t, path)


def test_generated_at_present(tmp_path):
    """Read generated_at from present metadata."""
    p = tmp_path / "catalog.parquet"
    _write(p, {b"sddb_generated_at": b"2026-10-06"})
    assert Catalog.from_file(p).generated_at == "2026-10-06"


def test_generated_at_absent_is_none(tmp_path):
    """Old releases without metadata return None gracefully."""
    p = tmp_path / "catalog.parquet"
    _write(p, {})
    assert Catalog.from_file(p).generated_at is None
