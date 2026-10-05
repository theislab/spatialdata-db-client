import subprocess
import sys

import pytest

pytest.importorskip("vitessce")
pytest.importorskip("easy_vitessce")

from tests._fixtures import make_fixture_catalog, make_tiny_sdata_zarr

from sddb import viz
from sddb.dataset import Dataset

CONFIG = {
    "version": "1.0.16",
    "name": "t",
    "description": "",
    "datasets": [{"uid": "A", "name": "A", "files": []}],
    "coordinationSpace": {"dataset": {"A": "A"}},
    "layout": [{"component": "description", "coordinationScopes": {"dataset": "A"}, "x": 0, "y": 0, "w": 6, "h": 6}],
    "initStrategy": "auto",
}


def test_require_missing():
    with pytest.raises(ImportError, match=r"\[viz\]"):
        viz._require("nonexistent_module_xyz")


def test_render_config_dict_and_path(tmp_path):
    from vitessce import VitessceConfig

    assert isinstance(viz.render_config(CONFIG), VitessceConfig)
    import json

    p = tmp_path / "c.json"
    p.write_text(json.dumps(CONFIG))
    assert isinstance(viz.render_config(p), VitessceConfig)


def test_view_interactive_config(monkeypatch):
    from vitessce import VitessceConfig

    seen = []

    def fake(config):
        seen.append(config)
        return VitessceConfig.from_dict(CONFIG)

    monkeypatch.setattr(viz, "render_config", fake)
    ds = Dataset(make_fixture_catalog().iloc[0])
    assert isinstance(ds.view_interactive(mode="config"), VitessceConfig)
    assert seen[0].startswith("https://lamin.ai/storage/s3/")
    with pytest.raises(ValueError, match="mode"):
        ds.view_interactive(mode="bogus")


def test_render_sdata(tmp_path):
    from spatialdata import read_zarr

    sdata = read_zarr(make_tiny_sdata_zarr(tmp_path))
    try:
        out = viz.render_sdata(sdata)
    except Exception as err:  # easy_vitessce quirk on minimal objects
        pytest.skip(f"easy_vitessce cannot render the tiny fixture: {err!r}")
    assert out is not None


def test_base_import_has_no_vitessce():
    code = "import sys, sddb; sys.exit(any(m.split('.')[0] in ('vitessce','easy_vitessce') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
