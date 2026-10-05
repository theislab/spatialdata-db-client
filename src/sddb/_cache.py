"""Cache directory resolution."""

from __future__ import annotations

import os
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
