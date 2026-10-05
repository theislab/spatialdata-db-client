"""Cache directory resolution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import platformdirs


def cache_dir(explicit: str | Path | None = None) -> Path:
    """Resolve and create the sddb cache directory.

    Precedence: ``explicit`` > ``SDDB_CACHE_DIR`` env var > ``platformdirs.user_cache_dir("sddb")``.

    Parameters
    ----------
    explicit
        Directory to use, overriding the environment and the default.

    Returns
    -------
    The (created) cache directory.
    """
    path = Path(explicit or os.environ.get("SDDB_CACHE_DIR") or platformdirs.user_cache_dir("sddb"))
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


_resolve_cache = cache_dir


def _local_path(url: str) -> Path | None:
    """Return the local filesystem path for a ``file://`` URL or plain path, else None."""
    if url.startswith("file://"):
        return Path(urllib.parse.unquote(urllib.parse.urlparse(url).path))
    if "://" not in url:
        return Path(url).expanduser()
    return None


def catalog_cache_path(url: str, *, cache_dir: str | Path | None = None) -> Path:
    """Return the cache file path a catalog ``url`` is stored at (without fetching)."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    name = Path(urllib.parse.urlparse(url).path).name or "catalog.parquet"
    folder = _resolve_cache(cache_dir) / "catalog"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}-{name}"


def _age(path: Path) -> str:
    seconds = max(0.0, time.time() - path.stat().st_mtime)
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def _http_fetch(url: str, dest: Path, etag_file: Path, *, refresh: bool) -> None:
    """Conditional GET ``url`` into ``dest``; a 304 leaves the cached file untouched.

    The sidecar records which validator was stored (``etag`` or ``last_modified``) so the matching
    conditional header (``If-None-Match`` / ``If-Modified-Since``) is sent next time.
    """
    from sddb import __version__

    headers = {"User-Agent": f"spatialdata-db-client/{__version__}"}
    if not refresh and dest.exists() and etag_file.exists():
        try:
            stored = json.loads(etag_file.read_text())
        except ValueError:
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
        if stored.get("etag"):
            headers["If-None-Match"] = stored["etag"]
        elif stored.get("last_modified"):
            headers["If-Modified-Since"] = stored["last_modified"]
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(dest)
            validator = {}
            if resp.headers.get("ETag"):
                validator["etag"] = resp.headers["ETag"]
            elif resp.headers.get("Last-Modified"):
                validator["last_modified"] = resp.headers["Last-Modified"]
            if validator:
                etag_file.write_text(json.dumps(validator))
            else:
                etag_file.unlink(missing_ok=True)
    except urllib.error.HTTPError as err:
        if err.code != 304:
            raise


def fetch_catalog(url: str, *, cache_dir: str | Path | None = None, refresh: bool = False) -> Path:
    """Fetch a catalog parquet from ``url`` into the cache and return the local path.

    HTTP(S): conditional GET using a stored ETag/Last-Modified; if the remote is unreachable and a
    cached copy exists, return the cache with a ``warnings.warn`` naming its age; if unreachable and no
    cache exists, raise. A ``file://`` or local path is used directly (copied into cache). ``refresh=True``
    bypasses the conditional check and re-downloads.

    Parameters
    ----------
    url
        ``http(s)://``, ``file://`` URL or local path of the catalog parquet.
    cache_dir
        Cache directory override.
    refresh
        Re-download regardless of the cached copy.

    Returns
    -------
    Path of the cached parquet.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if "://" in url and scheme not in {"http", "https", "file"}:
        raise ValueError(
            f"unsupported catalog URL scheme {scheme!r} in {url!r}; use http(s)://, file:// or a local path"
        )
    dest = catalog_cache_path(url, cache_dir=cache_dir)
    etag_file = dest.with_suffix(dest.suffix + ".etag")
    try:
        local = _local_path(url)
        if local is not None:
            if not local.is_file():
                raise FileNotFoundError(f"catalog not found: {url}")
            if refresh or not dest.exists() or dest.stat().st_mtime_ns != local.stat().st_mtime_ns:
                shutil.copy2(local, dest)
        else:
            _http_fetch(url, dest, etag_file, refresh=refresh)
    except (OSError, urllib.error.URLError) as err:
        if dest.exists():
            warnings.warn(
                f"could not reach {url} ({err}); using cached catalog that is {_age(dest)} old",
                stacklevel=2,
            )
            return dest
        raise FileNotFoundError(f"could not fetch catalog from {url} and no cached copy exists: {err}") from err
    return dest
