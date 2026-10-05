"""Optional interactive visualisation (the ``[viz]`` extra): Vitessce widgets in notebooks."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialdata import SpatialData

_RENDER = {
    "images": "render_images",
    "labels": "render_labels",
    "shapes": "render_shapes",
    "points": "render_points",
}


def _require(mod: str) -> Any:
    """Import ``mod`` or raise an ImportError naming the ``[viz]`` extra."""
    try:
        return importlib.import_module(mod)
    except ImportError as err:
        raise ImportError(f"{mod!r} is missing: install spatialdata-db[viz] for interactive viz") from err


def render_config(config: dict[str, Any] | str | Path) -> Any:
    """Build a ``VitessceConfig`` from a published per-dataset config.

    Parameters
    ----------
    config
        The config dict, or a local path / ``http(s)``/``file`` URL of the config JSON.

    Returns
    -------
    A ``vitessce.VitessceConfig``; call ``.widget()`` on it to display.
    """
    vitessce = _require("vitessce")
    if not isinstance(config, dict):
        config = _read_json(str(config))
    return vitessce.VitessceConfig.from_dict(config)


def _read_json(src: str) -> dict[str, Any]:
    if src.startswith(("http://", "https://", "file://")):
        import urllib.request

        with urllib.request.urlopen(src, timeout=30) as resp:
            data: dict[str, Any] = json.load(resp)
        return data
    loaded: dict[str, Any] = json.loads(Path(src).read_text())
    return loaded


def render_sdata(sdata: SpatialData, **kwargs: Any) -> Any:
    """Interactive Vitessce view of a loaded SpatialData via ``easy_vitessce``.

    Every image, label, shape and point element is added as a layer (``kwargs`` are passed to
    ``.pl.show()``). Returns the widget.
    """
    _require("easy_vitessce")  # enables the Vitessce-backed ``sdata.pl`` accessor
    _require("spatialdata_plot")
    for kind, method in _RENDER.items():
        for name in getattr(sdata, kind):
            getattr(sdata.pl, method)(name)  # accumulates layers on the accessor
    return sdata.pl.show(**kwargs)
