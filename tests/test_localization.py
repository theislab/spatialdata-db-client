import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from anndata import AnnData
from geopandas import GeoDataFrame
from shapely.geometry import Point
from spatialdata.models import Image2DModel, ShapesModel, TableModel

from sddb import adapter, remote


def _shapes(n):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (n, 2))
    return ShapesModel.parse(GeoDataFrame({"geometry": [Point(*p) for p in xy], "radius": 1.0}))


def _table(n, region):
    ad = AnnData(
        X=np.random.default_rng(1).poisson(1.0, (n, 12)).astype("float32"),
        obs=pd.DataFrame({"region": pd.Categorical([region] * n), "instance_id": np.arange(n)}),
    )
    return TableModel.parse(ad, region=region, region_key="region", instance_key="instance_id")


@pytest.fixture
def vhd_store(tmp_path) -> Path:
    rng = np.random.default_rng(2)
    imgs = {
        "s_full_image": Image2DModel.parse(rng.integers(0, 255, (3, 128, 128), dtype="uint8")),
        "s_hires_image": Image2DModel.parse(rng.integers(0, 255, (3, 32, 32), dtype="uint8")),
    }
    shapes, tables = {}, {}
    for um, n in [("002um", 400), ("008um", 100), ("016um", 25)]:
        sname = f"s_square_{um}"
        shapes[sname] = _shapes(n)
        tables[f"square_{um}"] = _table(n, sname)
    store = tmp_path / "vhd.zarr"
    sd.SpatialData(images=imgs, shapes=shapes, tables=tables).write(store)
    return store


KEEP = ["images/s_full_image", "tables/square_008um", "shapes/s_square_008um"]


def test_open_elements_subset(vhd_store, tmp_path):
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=tmp_path / "cache")
    assert set(out.images) == {"s_full_image"}
    assert set(out.tables) == {"square_008um"}
    assert set(out.shapes) == {"s_square_008um"}
    assert out.tables["square_008um"].n_obs == 100


def test_open_elements_prunes_consolidated(vhd_store, tmp_path):
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=tmp_path / "c")
    assert "s_hires_image" not in out.images
    (loc,) = (tmp_path / "c").glob("vhd__*.zarr")
    keys = set(json.loads((loc / "zarr.json").read_text())["consolidated_metadata"]["metadata"])
    dropped = ("images/s_hires_image", "tables/square_002um", "shapes/s_square_016um")
    assert not [k for k in keys if k.startswith(dropped)]
    assert any(k.startswith("tables/square_008um/") for k in keys)


def test_open_elements_missing_path_raises(vhd_store, tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        remote.open_sdata(str(vhd_store), lazy=False, elements=["tables/nope"], cache_dir=tmp_path / "c")


def test_open_elements_empty_raises(vhd_store, tmp_path):
    with pytest.raises(ValueError, match="element paths"):
        remote.open_sdata(str(vhd_store), lazy=False, elements=[], cache_dir=tmp_path / "c")


def test_open_elements_lazy_rejected(vhd_store, tmp_path):
    with pytest.raises(NotImplementedError, match="lazy=False"):
        remote.open_sdata(str(vhd_store), lazy=True, elements=KEEP, cache_dir=tmp_path / "c")


def test_prune_consolidated_noop_without_block(tmp_path):
    zj = tmp_path / "zarr.json"
    zj.write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
    before = zj.read_text()
    remote._prune_consolidated(tmp_path, {"images"})
    assert zj.read_text() == before


def test_cache_full_then_partial_no_collision(vhd_store, tmp_path):
    cache = tmp_path / "cache"
    full = remote.open_sdata(str(vhd_store), lazy=False, cache_dir=cache)
    assert "square_002um" in full.tables
    part = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=cache)
    assert set(part.tables) == {"square_008um"}
    dirs = sorted(p.name for p in cache.glob("*.zarr"))
    assert len(dirs) == 2
    assert any("__" in d for d in dirs)


def test_partial_copy_atomic_on_failure(vhd_store, tmp_path, monkeypatch):
    cache = tmp_path / "cache"

    def boom(dest, kept):
        raise RuntimeError("simulated mid-build failure")

    monkeypatch.setattr(remote, "_prune_consolidated", boom)
    with pytest.raises(FileNotFoundError, match="simulated"):  # wrapped per the open_sdata contract
        remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=cache)
    assert not list(cache.glob("vhd__*.zarr"))
    assert not list(cache.glob("vhd.partial-*"))
    monkeypatch.undo()
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=cache)
    assert set(out.tables) == {"square_008um"}


def test_prune_zmetadata_v2(tmp_path):
    import json

    from sddb.remote import _prune_consolidated

    dest = tmp_path / "v2.zarr"
    (dest / "images" / "keep" / "s0").mkdir(parents=True)
    (dest / "images" / "drop").mkdir(parents=True)
    zmeta = {
        "zarr_consolidated_format": 1,
        "metadata": {
            ".zgroup": {"zarr_format": 2},
            "images/.zgroup": {"zarr_format": 2},
            "images/keep/.zgroup": {"zarr_format": 2},
            "images/keep/s0/.zarray": {"shape": [3, 4, 4]},
            "images/drop/.zgroup": {"zarr_format": 2},
            "images/drop/.zarray": {"shape": [3, 2, 2]},
        },
    }
    (dest / ".zmetadata").write_text(json.dumps(zmeta))
    _prune_consolidated(dest, {"images", "images/keep"})
    out = json.loads((dest / ".zmetadata").read_text())["metadata"]
    assert "images/keep/s0/.zarray" in out
    assert not any(k.startswith("images/drop") for k in out)
    assert "images/.zgroup" in out
    assert ".zgroup" in out


def test_prune_zmetadata_unknown_form_rejected(tmp_path):
    import pytest

    from sddb.remote import _prune_consolidated

    dest = tmp_path / "bad.zarr"
    dest.mkdir()
    (dest / ".zmetadata").write_text('{"weird": 1}')
    with pytest.raises(NotImplementedError):
        _prune_consolidated(dest, {"images", "images/keep"})


def test_region_closure_autoincludes(vhd_store, tmp_path):
    # request ONLY the 008um table; its region shapes must be auto-included so the annotation resolves
    out = remote.open_sdata(str(vhd_store), lazy=False, elements=["tables/square_008um"], cache_dir=tmp_path / "c")
    assert "s_square_008um" in out.shapes  # auto-included
    assert out.tables["square_008um"].obs["region"].cat.categories.tolist() == ["s_square_008um"]


def test_resolve_elements_one_each(vhd_store):
    roles = [
        {"role": "image", "select": "full_image"},
        {"role": "table", "select": "square_008um"},
        {"role": "shapes", "select": "square_008um"},
    ]
    got = adapter._resolve_elements(str(vhd_store), roles)
    assert sorted(got) == ["images/s_full_image", "shapes/s_square_008um", "tables/square_008um"]


def test_resolve_elements_suffix_rejects_sibling(tmp_path):
    # suffix "square_016um" must NOT also match "square_016um_for_vitessce"
    store = tmp_path / "s.zarr"
    sd.SpatialData(
        shapes={
            "a_square_016um": _shapes(5),
            "a_square_016um_for_vitessce": _shapes(5),
        }
    ).write(store)
    got = adapter._resolve_elements(str(store), [{"role": "shapes", "select": "square_016um"}])
    assert got == ["shapes/a_square_016um"]


def test_resolve_elements_ambiguous_raises(vhd_store):
    with pytest.raises(ValueError, match="matched 3"):
        adapter._resolve_elements(str(vhd_store), [{"role": "table", "select": "um", "match": "substring"}])


def test_resolve_elements_missing_raises(vhd_store):
    with pytest.raises(ValueError, match="no element"):
        adapter._resolve_elements(str(vhd_store), [{"role": "image", "select": "nope"}])


class _FakeAdapter:
    name, version, config = "fake", "0", {}

    def requirements(self):
        return {"elements": [
            {"role": "image", "select": "full_image"},
            {"role": "table", "select": "square_008um"},
            {"role": "shapes", "select": "square_008um"},
        ]}

    def validate_meta(self, meta):
        return []

    def validate(self, sdata):
        return [] if set(sdata.tables) == {"square_008um"} else ["wrong tables"]

    def build(self, sdata):
        return sorted(sdata.tables)


def _members_df(vhd_store):
    return pd.DataFrame([{
        "uid": "u1", "zarr_url": str(vhd_store), "lamin_version": "1", "fingerprint": "fp",
    }])


def test_build_localizes_from_requirements(vhd_store, tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    from sddb import adapter
    view = adapter.build(_members_df(vhd_store), _FakeAdapter(), verify=False)
    uid, thunk = view[0]
    assert thunk() == ["square_008um"]


def test_build_provenance_records_elements(vhd_store, tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "prov"
    adapter.build(_members_df(vhd_store), _FakeAdapter(), materialize=True, out=out, verify=False)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["requirements"]["elements"][0]["role"] == "image"
    member = prov["members"][0]
    assert member["elements"] == ["images/s_full_image", "shapes/s_square_008um", "tables/square_008um"]


class _Fake002Adapter(_FakeAdapter):
    def requirements(self):
        return {"elements": [
            {"role": "image", "select": "full_image"},
            {"role": "table", "select": "square_002um"},
            {"role": "shapes", "select": "square_002um"},
        ]}

    def validate(self, sdata):
        return []


def test_provenance_differs_by_element_selector(vhd_store, tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    members = _members_df(vhd_store)
    out_a, out_b = tmp_path / "prov_a", tmp_path / "prov_b"
    adapter.build(members, _FakeAdapter(), materialize=True, out=out_a, verify=False)
    adapter.build(members, _Fake002Adapter(), materialize=True, out=out_b, verify=False)
    a = json.loads((out_a / "provenance.json").read_text())
    b = json.loads((out_b / "provenance.json").read_text())
    assert a["members"][0]["elements"] == ["images/s_full_image", "shapes/s_square_008um", "tables/square_008um"]
    assert b["members"][0]["elements"] == ["images/s_full_image", "shapes/s_square_002um", "tables/square_002um"]
    assert a["members"][0]["elements"] != b["members"][0]["elements"]
    assert a["requirements"]["elements"] != b["requirements"]["elements"]


def test_close_region_tolerates_corrupt_table_metadata(tmp_path):
    import fsspec

    (tmp_path / "tables" / "t").mkdir(parents=True)
    (tmp_path / "tables" / "t" / "zarr.json").write_text("{not json")
    fs = fsspec.filesystem("file")
    assert remote._close_region(fs, str(tmp_path), ["tables/t"]) == ["tables/t"]


def test_open_member_verifies_then_opens(vhd_store, tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    import sddb.freeze as fz
    import sddb.remote as rem
    from sddb.dataset import Source
    from sddb.freeze import FrozenCohort

    members = _members_df(vhd_store)
    fc = FrozenCohort("1", "2026-10-08T00:00:00Z", Source(url=None, version=None), {}, "sha256:x", members)

    calls = []
    monkeypatch.setattr(fz, "verify_members", lambda m, **k: calls.append("verify"))
    _orig = rem.open_sdata

    def spy(*a, **k):
        calls.append("open")
        return _orig(*a, **k)

    monkeypatch.setattr(rem, "open_sdata", spy)

    sdata = fc.open_member("u1", elements=["tables/square_008um"])
    assert calls == ["verify", "open"]
    assert set(sdata.tables) == {"square_008um"}


@pytest.mark.network
def test_localize_oliveira_excludes_2um(tmp_path):
    # Real Oliveira Visium HD store: localizing 8um + H&E must not download the 8.1M-row 2um table.
    import sddb

    cat = sddb.Catalog()
    url = cat.to_df().set_index("uid").loc["4OgPqTscrYc4veqE0000", "zarr_url"]
    roles = [
        {"role": "image", "select": "full_image"},
        {"role": "table", "select": "square_008um"},
        {"role": "shapes", "select": "square_008um"},
    ]
    paths = adapter._resolve_elements(url, roles)
    sdata = remote.open_sdata(url, lazy=False, elements=paths, cache_dir=tmp_path)
    assert "square_002um" not in sdata.tables
    assert any("square_008um" in t for t in sdata.tables)
    cached = next(tmp_path.glob("*.zarr"))
    assert not list(cached.glob("tables/*002um*"))  # 2um subtree never copied


def test_same_basename_stores_get_distinct_cache_dirs(tmp_path):
    cache = tmp_path / "cache"
    got = {}
    for tag, n in (("a", 5), ("b", 9)):
        store = tmp_path / tag / "data.zarr"
        sd.SpatialData(shapes={"s": _shapes(n)}).write(store)
        got[tag] = remote.open_sdata(str(store), lazy=False, elements=["shapes/s"], cache_dir=cache)
    assert len(got["a"].shapes["s"]) == 5
    assert len(got["b"].shapes["s"]) == 9
    assert len(list(cache.glob("data__*.zarr"))) == 2


def test_open_elements_partial_store_error_names_url(vhd_store, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad read")

    monkeypatch.setattr(sd, "read_zarr", boom)
    with pytest.raises(FileNotFoundError, match="Cannot open SpatialData zarr.*bad read"):
        remote.open_sdata(str(vhd_store), lazy=False, elements=KEEP, cache_dir=tmp_path / "c")


def test_resolve_elements_role_canonicalization(vhd_store):
    url = str(vhd_store)
    for role, want in [
        ("shape", "shapes/s_square_008um"),
        ("shapes", "shapes/s_square_008um"),
        ("image", "images/s_full_image"),
        ("images", "images/s_full_image"),
        ("table", "tables/square_008um"),
    ]:
        select = "full_image" if "image" in role else "square_008um"
        assert adapter._resolve_elements(url, [{"role": role, "select": select}]) == [want]


class _NoRequirementsAdapter:
    name, version, config = "noreq", "0", {}

    def validate_meta(self, meta):
        return []

    def validate(self, sdata):
        return []

    def build(self, sdata):
        return sorted(sdata.tables)


def test_open_adapter_without_requirements(vhd_store, tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    sdata = adapter._open(str(vhd_store), _NoRequirementsAdapter())
    assert "square_002um" in sdata.tables  # whole store


def test_materialize_adapter_without_requirements(vhd_store, tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "prov"
    adapter.build(_members_df(vhd_store), _NoRequirementsAdapter(), materialize=True, out=out, verify=False)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["requirements"] == {}
    assert prov["members"][0]["elements"] is None


class _FakeTableOnlyAdapter(_FakeAdapter):
    def requirements(self):
        return {"elements": [{"role": "table", "select": "square_008um"}]}

    def validate(self, sdata):
        return []


def test_provenance_reflects_region_closure(vhd_store, tmp_path, monkeypatch):
    monkeypatch.setenv("SDDB_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "prov"
    adapter.build(_members_df(vhd_store), _FakeTableOnlyAdapter(), materialize=True, out=out, verify=False)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["members"][0]["elements"] == ["shapes/s_square_008um", "tables/square_008um"]
