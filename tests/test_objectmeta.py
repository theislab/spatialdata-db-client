from __future__ import annotations

import pytest
from tests._fixtures import make_tiny_sdata_zarr

from sddb.objectmeta import ObjectMetaError, compute_fingerprint, fetch_many, fetch_object_metadata


def test_fetch_reads_metadata_only(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    m = fetch_object_metadata("u1", str(z))
    assert m.uid == "u1"
    assert "tables/table" in m.elements
    assert m.tables == ("table",)
    assert m.fingerprint.startswith("sha256:")


def test_fingerprint_changes_with_shape(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    base = fetch_object_metadata("u1", str(z)).fingerprint
    mutated = compute_fingerprint(
        {"tables/table": {"type": "tables", "shape": (5, 3), "dtype": "float32"}}, ("table",), None
    )
    assert base != mutated


def test_fingerprint_changes_with_root_provenance():
    el = {"type": "tables", "shape": (4, 3), "dtype": "float32"}

    def fp(stamp: str) -> str:
        root = {"sddb_provenance": {"converted_at": stamp}}
        return compute_fingerprint({"tables/table": el, "__root__": root}, ("table",), None)

    assert fp("A") != fp("B")
    assert fp("A") == fp("A")


def test_fetch_many_all_or_nothing(tmp_path):
    z = make_tiny_sdata_zarr(tmp_path)
    with pytest.raises(ObjectMetaError) as ei:
        fetch_many([("good", str(z)), ("bad", str(tmp_path / "missing.zarr"))], workers=2)
    assert "bad" in ei.value.failed
