from __future__ import annotations

import io
import json

import pytest

from sddb import _cache


class _Resp(io.BytesIO):
    headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_legacy_sidecar_and_user_agent(tmp_path, monkeypatch):
    dest = tmp_path / "c.parquet"
    dest.write_bytes(b"old")
    etag = tmp_path / "c.parquet.etag"
    etag.write_text(json.dumps(["legacy", "list"]))
    seen = {}

    def fake_urlopen(req, timeout):
        seen.update(req.headers)
        return _Resp(b"new")

    monkeypatch.setattr(_cache.urllib.request, "urlopen", fake_urlopen)
    _cache._http_fetch("https://x/c.parquet", dest, etag, refresh=False)
    assert dest.read_bytes() == b"new"
    assert "If-none-match" not in seen
    assert seen["User-agent"].startswith("spatialdata-db-client/")


@pytest.mark.parametrize("url", ["s3://b/c.parquet", "ftp://h/c.parquet"])
def test_unsupported_scheme(url, tmp_path):
    with pytest.raises(ValueError, match=url.split(":")[0]):
        _cache.fetch_catalog(url, cache_dir=tmp_path)
