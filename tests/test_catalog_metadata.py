"""Test Catalog metadata reading (generated_at from release stamping)."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sddb import Catalog
from sddb import _catalog_schema as schema


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


def _release_info_meta(version):
    return {b"sddb_release_info": json.dumps({"catalog_schema_version": version}).encode()}


def test_schema_version_supported_loads(tmp_path):
    """A catalog declaring the supported schema version loads."""
    p = tmp_path / "catalog.parquet"
    _write(p, _release_info_meta(schema.SUPPORTED_CATALOG_SCHEMA))
    assert len(Catalog.from_file(p)) == 1


def test_schema_version_absent_is_legacy(tmp_path):
    """A legacy catalog without a declared schema version loads (missing == OK)."""
    p = tmp_path / "catalog.parquet"
    _write(p, {})  # no sddb_release_info at all
    assert len(Catalog.from_file(p)) == 1
    p2 = tmp_path / "catalog2.parquet"
    _write(p2, {b"sddb_release_info": json.dumps({"hub_sha": "abc"}).encode()})  # provenance, no version key
    assert len(Catalog.from_file(p2)) == 1


def test_schema_version_too_new_raises(tmp_path):
    """A catalog declaring a newer schema in its parquet metadata is refused with a clear error."""
    p = tmp_path / "catalog.parquet"
    _write(p, _release_info_meta(schema.SUPPORTED_CATALOG_SCHEMA + 1))
    with pytest.raises(ValueError, match=f"schema version {schema.SUPPORTED_CATALOG_SCHEMA + 1}"):
        Catalog.from_file(p)


def test_schema_version_broken_provenance_raises(tmp_path):
    """Corrupt sddb_release_info is a bug, not a legacy catalog — it must fail closed."""
    p = tmp_path / "catalog.parquet"
    _write(p, {b"sddb_release_info": b"{not json"})
    with pytest.raises(ValueError, match="not valid JSON"):
        Catalog.from_file(p)


def test_schema_version_non_integer_raises(tmp_path):
    """A present catalog_schema_version must be an integer (a string or float fails closed)."""
    for bad in ("3", 2.5):
        p = tmp_path / f"catalog_{bad}.parquet"
        _write(p, _release_info_meta(bad))
        with pytest.raises(ValueError, match="must be an integer"):
            Catalog.from_file(p)
