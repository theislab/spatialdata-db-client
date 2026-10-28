from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path_factory, monkeypatch):
    """Keep fetches out of the real user cache; a test may still set ``SDDB_CACHE_DIR`` itself."""
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path_factory.mktemp("sddb-cache")))
