"""Test remote-lazy fallback error message."""

from __future__ import annotations

import pytest
from tests._fixtures import make_fixture_catalog

from sddb.dataset import Results


def test_load_remote_lazy_gives_helpful_error(monkeypatch):
    """Test that Dataset.load(lazy=True) on a remote URL gives a helpful error mentioning .download() and lazy=False."""
    # Monkeypatch remote.open_sdata to raise NotImplementedError like upstream spatialdata would
    def mock_open_sdata(zarr_url: str, *, lazy: bool = True, cache_dir=None):
        if lazy and zarr_url.startswith(("s3://", "http://", "https://")):
            raise NotImplementedError("Remote lazy/partial open is not supported")
        raise AssertionError(f"Unexpected call to mock_open_sdata: lazy={lazy}, zarr_url={zarr_url}")

    import sddb.remote
    monkeypatch.setattr(sddb.remote, "open_sdata", mock_open_sdata)

    # Create a dataset with a remote zarr URL
    df = make_fixture_catalog().iloc[:1].copy()
    df["zarr_url"] = "s3://bucket/some.zarr"
    d = Results(df)[0]

    # Call load(lazy=True) on the remote dataset
    with pytest.raises(NotImplementedError) as exc_info:
        d.load(lazy=True)

    # Check that the error message is helpful
    error_msg = str(exc_info.value)
    assert ".download(" in error_msg, f"Error message missing '.download(': {error_msg}"
    assert "lazy=False" in error_msg, f"Error message missing 'lazy=False': {error_msg}"
