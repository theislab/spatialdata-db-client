"""``materialize``: a synthetic cohort written as a loader collection, rerun and read back."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

# the loader extra
pytest.importorskip("pydantic")
pytest.importorskip("squidpy")
pytest.importorskip("dask_image")
pytest.importorskip("spatialbatch")
import shapely
import spatialdata as sd
import zarr
from anndata.io import read_elem
from scipy import ndimage
from spatialdata.models import TableModel
from tests.test_loader_adapters import he_image, make_sdata, squares

from sddb.contract import DatasetContract, validate
from sddb.loader import (
    Centroids,
    Rasterize,
    StainNormalize,
    TissueMask,
    _defaults,
    _materialize,
    default_chain,
    materialize,
)

ROLES = {"image": "^images/he$", "table": "^tables/table$"}
SIZE = 96
# Gene Expression features with an Ensembl id; uidB also measures GENE5
GENES = {f"GENE{i}": f"ENSG{i:011d}" for i in range(1, 6)}


class FakeMember:
    """The part of :class:`sddb.Dataset` that ``materialize`` uses: uid, study_id, catalog columns, ``to_sdata``."""

    def __init__(self, uid, path, tissue="breast", **columns):
        self.uid, self.study_id, self.tissue, self._path = uid, f"study_{uid}", tissue, path
        self.__dict__.update(columns)  # further catalog columns, e.g. technology, hne_image, mpp

    def to_sdata(self):
        return sd.read_zarr(self._path)

    def _local_store(self):
        return self._path


class FakeCohort(list):
    _source = SimpleNamespace(release="2026-11-01")


def write_member(tmp_path, uid, *, seed, extra_gene=False, tiny=False, labels=False, dup_id=False, pixel_size=None):
    """A member store: H&E tissue disc, square cells, a table keyed by symbol with Ensembl ids and controls.

    `dup_id` adds a Gene Expression row ``GENE1B`` with GENE1's Ensembl id. `pixel_size` is recorded in the store's
    attrs as spatialdata-io's Atera reader does (microns per ``global`` unit; the image is the identity onto it).
    """
    img = he_image(SIZE, SIZE, seed=seed, disk=(SIZE / 2, SIZE / 2, SIZE * 0.4))
    xy = [(x, y) for x in range(20, 80, 12) for y in range(20, 80, 12)]
    shapes = squares(xy, 6)
    if tiny:  # a cell between the squares that covers no pixel centre, on the source grid or the written one
        shapes = pd.concat([shapes, gpd.GeoDataFrame(geometry=[shapely.box(50.1, 50.1, 50.3, 50.3)], index=["tiny"])])
    if labels:  # the cells as a raster, plus two 1 px cells: at 0.5 um/px only one of them keeps a pixel
        raster = np.asarray(_rasterize(shapes, SIZE))
        raster[50, 50], raster[51, 51] = raster.max() + 1, raster.max() + 2
        sdata = make_sdata(img, labels=raster, annotate="cell_labels", image_chunks=32)
        ids = np.asarray(sdata.tables["table"].obs["cell_id"])
        # obsm["spatial"] in the labels' intrinsic coordinates, where pixel j has its centre at j + 0.5
        com = np.asarray(ndimage.center_of_mass(np.ones_like(raster), raster, ids))[:, ::-1] + 0.5
    else:
        sdata = make_sdata(img, shapes=shapes, image_chunks=32)
    genes = dict(GENES) if extra_gene else {k: v for k, v in GENES.items() if k != "GENE5"}
    if dup_id:
        genes["GENE1B"] = GENES["GENE1"]
    var = pd.DataFrame(
        {
            "gene_ids": [*genes.values(), "", "NegCtrl_1", "NegCtrl_2"],
            "feature_types": ["Gene Expression"] * (len(genes) + 1) + ["Negative Control Probe"] * 2,
        },
        index=[*genes, "NOENSEMBL", "NegCtrl_1", "NegCtrl_2"],
    )
    old = sdata.tables["table"]
    counts = np.random.default_rng(seed).poisson(2.0, (old.n_obs, len(var))).astype(np.int32)
    table = ad.AnnData(X=counts, obs=old.obs.copy(), var=var)
    del sdata.tables["table"]
    region = "cell_labels" if labels else "cells"
    if labels:
        table.obsm["spatial"] = com
    sdata.tables["table"] = TableModel.parse(table, region=region, region_key="region", instance_key="cell_id")
    if pixel_size is not None:
        sdata.attrs["pixel_size"] = pixel_size
    path = tmp_path / f"{uid}.zarr"
    sdata.write(path)
    return path, img, counts, var


@pytest.fixture
def cohort(tmp_path):
    a = write_member(tmp_path, "uidA", seed=0)
    b = write_member(tmp_path, "uidB", seed=1, extra_gene=True)
    # given in reverse uid order: datasets are written in uid order
    members = FakeCohort([FakeMember("uidB", b[0]), FakeMember("uidA", a[0])])
    return members, {"uidA": a, "uidB": b}


def _contract(dest, key):
    return DatasetContract.from_uns(read_elem(zarr.open_group(dest / key, mode="r")["uns"]["sddb_contract"]))


def _table(dest, key):
    """The written table (the dataset group also holds images and labels, which ``anndata.io.read_zarr`` refuses)."""
    group = zarr.open_group(dest / key, mode="r")
    return ad.AnnData(X=read_elem(group["X"]), obs=read_elem(group["obs"]), var=read_elem(group["var"]))


def test_materialize_synthetic_cohort_validates(cohort, tmp_path, caplog):
    members, sources = cohort
    dest = tmp_path / "collection.zarr"
    state = tmp_path / "collection.zarr.sddb"
    out = materialize(
        members,
        dest,
        adapters=default_chain(0.5),
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25, "uidB": 0.25},
        obs_columns=["tissue"],
    )
    assert out == dest
    assert validate(dest) == []
    root = zarr.open_group(dest, mode="r")
    assert sorted(k for k in root.group_keys() if k.startswith("dataset_")) == ["dataset_0", "dataset_1"]

    # the union, sorted: GENE5 only in uidB; NOENSEMBL (Gene Expression without an Ensembl id) kept by its name
    union = [GENES[f"GENE{i}"] for i in range(1, 6)] + ["NOENSEMBL", "NegCtrl_1", "NegCtrl_2"]
    references = []
    for key, uid in (("dataset_0", "uidA"), ("dataset_1", "uidB")):
        contract = _contract(dest, key)
        assert contract.uid == uid
        assert contract.study_id == f"study_{uid}"
        assert contract.catalog_release == "2026-11-01"
        assert contract.grid == "target"
        assert (contract.mpp_target, contract.mpp_native) == (0.5, 0.25)
        assert [a.name for a in contract.adapters] == [
            "GeneKey",
            "Rasterize",
            "Centroids",
            "TissueMask",
            "StainNormalize",
        ]
        genes = contract.adapters[0].params
        assert genes["join"] == "outer"
        assert genes["no_ensembl_id"] == 1
        assert genes["filled"] == (1 if uid == "uidA" else 0)
        assert (contract.n_features_present, contract.n_features_filled) == ((7, 1) if uid == "uidA" else (8, 0))
        references.append(contract.stain_reference)

        table = _table(dest, key)
        assert list(table.var_names) == union
        assert table.var["feature_types"].tolist() == ["Gene Expression"] * 6 + ["Negative Control Probe"] * 2
        assert table.var["present"].tolist() == [True] * 4 + [uid == "uidB"] + [True] * 3
        assert table.var["gene_ids"].isna().tolist() == [False] * 5 + [True] * 3
        assert set(table.obs["uid"]) == {uid}
        assert set(table.obs["tissue"]) == {"breast"}
        _, _, counts, var = sources[uid]
        # X holds the source counts of every feature, row for row by label id, and 0 for the one not measured
        source_rows = table.obs["label_id"].to_numpy() - 1
        expected = np.zeros((len(source_rows), len(union)), counts.dtype)
        for j, name in enumerate(union):
            if len(hit := np.flatnonzero((var["gene_ids"].to_numpy() == name) | (var.index == name))):
                expected[:, j] = counts[source_rows, hit[0]]
        np.testing.assert_array_equal(
            np.asarray(table.X.todense() if hasattr(table.X, "todense") else table.X), expected
        )
        assert (state / "done" / uid).is_file()
        assert json.loads((state / "done" / uid).read_text())["key"] == key
    assert references[0] is not None
    assert references[0] == references[1]  # one pooled reference for the cohort
    assert "uidA: 1 of 8 features of the collection are not measured, set to 0" in caplog.text
    assert not (state / "scratch").exists()
    assert sorted(p.name for p in dest.iterdir() if not p.name.startswith("zarr.json")) == ["dataset_0", "dataset_1"]


def test_rerun_skips_done_members(cohort, tmp_path, monkeypatch):
    members, _ = cohort
    dest = tmp_path / "collection.zarr"
    state = tmp_path / "collection.zarr.sddb"
    kwargs = {"role": "he", "roles": ROLES, "mpp_native": {"uidA": 0.25, "uidB": 0.25}}
    materialize(members, dest, adapters=default_chain(0.5), **kwargs)
    reference = _contract(dest, "dataset_0").stain_reference

    written = []
    write = _materialize.write_collection_dataset
    monkeypatch.setattr(
        _materialize,
        "write_collection_dataset",
        lambda group, *, key, **kw: written.append(key) or write(group, key=key, **kw),
    )
    materialize(members, dest, adapters=default_chain(0.5), **kwargs)
    assert written == []

    # a member without its marker is written again, under the same key, with the cohort's stain reference
    (state / "done" / "uidB").unlink()
    materialize(members, dest, adapters=default_chain(0.5), **kwargs)
    assert written == ["dataset_1"]
    assert _contract(dest, "dataset_1").stain_reference == reference
    root = zarr.open_group(dest, mode="r")
    assert sorted(k for k in root.group_keys() if k.startswith("dataset_")) == ["dataset_0", "dataset_1"]
    assert validate(dest) == []


def test_rerun_refuses_a_changed_cohort(cohort, tmp_path):
    members, _ = cohort
    dest = tmp_path / "collection.zarr"
    tmp_path / "collection.zarr.sddb"
    chain = [Rasterize(0.5), Centroids()]
    kwargs = {"role": "he", "roles": ROLES, "mpp_native": {"uidA": 0.25, "uidB": 0.25}}
    materialize(members, dest, adapters=chain, **kwargs)
    with pytest.raises(ValueError, match="dataset_1"):
        materialize(FakeCohort([m for m in members if m.uid == "uidA"]), dest, adapters=chain, **kwargs)


def test_mixed_grids_refused_at_collection(cohort, tmp_path):
    members, _ = cohort
    dest = tmp_path / "collection.zarr"
    tmp_path / "collection.zarr.sddb"
    kwargs = {"adapters": [Rasterize(0.5), Centroids()], "role": "he", "roles": ROLES, "mpp_native": {"uidA": 0.25}}
    with pytest.raises(ValueError, match=r"mixes.*target.*uidA.*native.*uidB"):
        materialize(members, dest, **kwargs)
    assert not any(dest.glob("dataset_*"))

    materialize(members, dest, allow_mixed_grids=True, **kwargs)
    assert validate(dest) == []
    assert _contract(dest, "dataset_0").grid == "target"
    assert _contract(dest, "dataset_1").grid == "native"
    assert _contract(dest, "dataset_1").allow_mixed_grids


def test_stain_needs_tissue_mask_first(cohort, tmp_path):
    members, _ = cohort
    with pytest.raises(ValueError, match="TissueMask"):
        materialize(
            members,
            tmp_path / "c.zarr",
            adapters=[Rasterize(None), Centroids(), StainNormalize()],
            role="he",
            roles=ROLES,
        )


def reference_tile(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Independent reference: the pixels under a tile centred on (cx, cy), zero-padded outside the array."""
    x0, y0 = int(np.floor(cx - size / 2 + 0.5)), int(np.floor(cy - size / 2 + 0.5))
    height, width = image.shape[-2:]
    out = np.zeros((*image.shape[:-2], size, size), dtype=image.dtype)
    cy0, cy1, cx0, cx1 = max(y0, 0), min(y0 + size, height), max(x0, 0), min(x0 + size, width)
    out[..., cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = image[..., cy0:cy1, cx0:cx1]
    return out


def test_loader_iterates_materialized(cohort, tmp_path):
    from spatialbatch import Loader
    from spatialbatch.io import SpatialDatasetCollection

    members, sources = cohort
    dest = tmp_path / "collection.zarr"
    tmp_path / "collection.zarr.sddb"
    # target mpp = native mpp: the written grid is the source grid, so tiles can be checked against the source image
    chain = [Rasterize(0.25), Centroids(), TissueMask()]
    materialize(members, dest, adapters=chain, role="he", roles=ROLES, mpp_native={"uidA": 0.25, "uidB": 0.25})

    root = zarr.open_group(dest, mode="r")
    written = {"uidA": root["dataset_0"], "uidB": root["dataset_1"]}
    for uid, group in written.items():
        np.testing.assert_array_equal(group["images"]["he"][...], sources[uid][1])

    loader = Loader(images=["he"], labels=["cells"], crop_px=32, batch_size=16).use_collection(
        SpatialDatasetCollection(dest)
    )
    seen = []
    for batch in loader:
        assert batch["images"]["he"].shape[1:] == (3, 32, 32)
        for i, (uid, (cx, cy)) in enumerate(zip(batch["obs"]["uid"], batch["coords"], strict=True)):
            np.testing.assert_array_equal(batch["images"]["he"][i], reference_tile(sources[uid][1], cx, cy, 32))
            labels = written[uid]["labels"]["cells"][...]
            np.testing.assert_array_equal(batch["labels"]["cells"][i], reference_tile(labels, cx, cy, 32))
            seen.append((uid, int(batch["obs"]["label_id"].iloc[i])))
    n = {
        uid: zarr.open_group(dest / key, mode="r")["obsm"]["spatial"].shape[0]
        for uid, key in (("uidA", "dataset_0"), ("uidB", "dataset_1"))
    }
    assert len(seen) == len(set(seen)) == n["uidA"] + n["uidB"] == 50


def test_cells_drawn_once(cohort, tmp_path, monkeypatch):
    from sddb.loader import _adapters

    calls = []
    draw = _adapters._draw_tile
    monkeypatch.setattr(_adapters, "_draw_tile", lambda *a, **k: calls.append(1) or draw(*a, **k))
    members, _ = cohort
    chain = [Rasterize(0.5), Centroids(), TissueMask()]
    materialize(
        members, tmp_path / "c.zarr", adapters=chain, role="he", roles=ROLES, mpp_native={"uidA": 0.25, "uidB": 0.25}
    )
    assert len(calls) == 2  # one tile per member: drawn for Centroids, then reread for TissueMask and the write


def test_cohort_method_on_local_stores(cohort, tmp_path):
    from sddb import SpatialDataCohort
    from sddb.dataset import Source

    _, sources = cohort
    df = pd.DataFrame(
        {
            "uid": ["uidA", "uidB"],
            "study_id": ["study_uidA", "study_uidB"],
            "zarr_url": [str(sources[u][0]) for u in ("uidA", "uidB")],
            "tissue": ["breast", "lung"],
        }
    )
    dest = SpatialDataCohort(df, source=Source(release="2026-11-01")).materialize(
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25, "uidB": 0.25},
        obs_columns=["tissue"],
    )
    assert validate(dest) == []
    assert set(_table(dest, "dataset_1").obs["tissue"]) == {"lung"}


def test_failed_member_not_marked_and_rewritten(cohort, tmp_path, monkeypatch):
    from sddb.contract import ContractError

    members, _ = cohort
    dest = tmp_path / "c.zarr"
    state = tmp_path / "c.zarr.sddb"
    kwargs = {"adapters": [Rasterize(0.5), Centroids()], "role": "he", "roles": ROLES}
    kwargs["mpp_native"] = {"uidA": 0.25, "uidB": 0.25}
    with monkeypatch.context() as m:
        m.setattr(_materialize, "validate_dataset", lambda group: ["synthetic issue"])
        with pytest.raises(ContractError, match=r"dataset_0 \(uidA\): synthetic issue"):
            materialize(members, dest, **kwargs)
    assert not (dest / "dataset_0").exists()  # a dataset that fails validation is removed
    assert not (state / "done").exists()

    # what an interrupted write leaves: a dataset dir without a done marker, replaced by the next run
    (dest / "dataset_1").mkdir()
    (dest / "dataset_1" / "partial").write_text("")
    materialize(members, dest, **kwargs)
    assert validate(dest) == []
    assert sorted(p.name for p in (state / "done").iterdir()) == ["uidA", "uidB"]


def test_collection_holds_only_zarr(cohort, tmp_path):
    import warnings

    from spatialbatch.io import SpatialDatasetCollection
    from zarr.errors import ZarrUserWarning

    members, _ = cohort
    dest = tmp_path / "c.zarr"
    chain = [Rasterize(0.5), Centroids()]
    materialize(members, dest, adapters=chain, role="he", roles=ROLES, mpp_native={"uidA": 0.25, "uidB": 0.25})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        assert validate(dest) == []
        assert len(list(SpatialDatasetCollection(dest))) == 2
        materialize(members, dest, adapters=chain, role="he", roles=ROLES, mpp_native={"uidA": 0.25, "uidB": 0.25})


def test_materialize_records_are_not_rebuilt(cohort, tmp_path):
    from sddb.loader import MATERIALIZE_RECORDS, adapter_from_record

    members, _ = cohort
    dest = tmp_path / "c.zarr"
    chain = [Rasterize(0.5), Centroids()]
    materialize(members, dest, adapters=chain, role="he", roles=ROLES, mpp_native={"uidA": 0.25, "uidB": 0.25})
    records = _contract(dest, "dataset_0").adapters
    assert MATERIALIZE_RECORDS == ("GeneKey",)
    assert records[0].name in MATERIALIZE_RECORDS
    with pytest.raises(ValueError, match="step of materialize.*MATERIALIZE_RECORDS"):
        adapter_from_record(records[0])
    rebuilt = [adapter_from_record(r) for r in records if r.name not in MATERIALIZE_RECORDS]
    assert [a.to_record() for a in rebuilt] == records[1:]
    with pytest.raises(ValueError, match="no loader adapter named 'Nope'"):
        adapter_from_record(records[1].model_copy(update={"name": "Nope"}))


def test_unreadable_marker_is_not_done(cohort, tmp_path, monkeypatch, caplog):
    members, _ = cohort
    dest, state = tmp_path / "c.zarr", tmp_path / "c.zarr.sddb"
    kwargs = {"adapters": [Rasterize(0.5), Centroids()], "role": "he", "roles": ROLES}
    kwargs["mpp_native"] = {"uidA": 0.25, "uidB": 0.25}
    materialize(members, dest, **kwargs)
    assert sorted(p.name for p in (state / "done").iterdir()) == ["uidA", "uidB"]  # no temp file left
    (state / "done" / "uidB").write_text("{trunc")

    written = []
    write = _materialize.write_collection_dataset
    monkeypatch.setattr(
        _materialize,
        "write_collection_dataset",
        lambda group, *, key, **kw: written.append(key) or write(group, key=key, **kw),
    )
    materialize(members, dest, **kwargs)
    assert written == ["dataset_1"]
    assert "unreadable" in caplog.text
    assert json.loads((state / "done" / "uidB").read_text())["key"] == "dataset_1"


def test_caller_chain_is_not_mutated(cohort, tmp_path, monkeypatch):
    import squidpy.experimental.im as sqim

    calls = []
    fit = sqim.fit_stain_reference
    monkeypatch.setattr(sqim, "fit_stain_reference", lambda *a, **k: calls.append(1) or fit(*a, **k))
    members, _ = cohort
    chain = default_chain(0.5)
    kwargs = {"role": "he", "roles": ROLES, "mpp_native": {"uidA": 0.25, "uidB": 0.25}}
    materialize(members, tmp_path / "a.zarr", adapters=chain, **kwargs)
    materialize(members, tmp_path / "b.zarr", adapters=chain, **kwargs)
    assert len(calls) == 2  # fit once per dest, not carried from the first into the second
    assert chain[-1].reference is None


@pytest.mark.parametrize(
    "changed",
    [
        [Rasterize(0.4), Centroids(), TissueMask()],
        [Rasterize(0.5), Centroids(), TissueMask(auto_max_pixels=1000)],
    ],
    ids=["rasterize-mpp", "tissue-kwarg"],
)
def test_rerun_with_another_chain_raises(cohort, tmp_path, monkeypatch, changed):
    members, _ = cohort
    dest, state = tmp_path / "c.zarr", tmp_path / "c.zarr.sddb"
    kwargs = {"role": "he", "roles": ROLES, "mpp_native": {"uidA": 0.25, "uidB": 0.25}}
    materialize(members, dest, adapters=[Rasterize(0.5), Centroids(), TissueMask()], **kwargs)
    (state / "done" / "uidB").unlink()

    written = []
    monkeypatch.setattr(_materialize, "write_collection_dataset", lambda *a, key, **kw: written.append(key))
    param = "mpp" if changed[0].mpp != 0.5 else "auto_max_pixels"
    with pytest.raises(ValueError, match=rf"dataset_0 ran (Rasterize|TissueMask) with {param}="):
        materialize(members, dest, adapters=changed, **kwargs)
    assert written == []


def test_cell_without_pixels_is_kept_and_counted_in_the_contract(tmp_path):
    path = write_member(tmp_path, "uidA", seed=0, tiny=True)[0]
    dest = materialize(
        FakeCohort([FakeMember("uidA", path)]),
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25},
    )
    assert validate(dest) == []
    contract = _contract(dest, "dataset_0")
    assert contract.n_centroid_fallback == 1
    group = zarr.open_group(dest / "dataset_0", mode="r")
    obs = read_elem(group["obs"])
    assert "tiny" in obs["cell_id"].tolist()
    assert len(obs) == len(sd.read_zarr(path).tables["table"])
    row = obs["cell_id"].tolist().index("tiny")
    # intrinsic (50.2, 50.2) at 0.25 um/px is (25.1, 25.1) at 0.5 um/px, minus 0.5 for the pixel-centre convention
    np.testing.assert_allclose(np.asarray(group["obsm"]["spatial"][row]), [24.6, 24.6], atol=1e-9)
    assert DatasetContract.from_uns(contract.to_uns()).n_centroid_fallback == 1


def test_cells_outside_the_image_are_kept_flagged_and_counted(tmp_path):
    """A rotated H&E: a cell over a fill corner of the grid and one beyond the grid keep their true centroids."""
    from spatialdata.transformations import Affine

    theta = np.deg2rad(20)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    corners = (rot @ np.array([[0, SIZE, SIZE, 0], [0, 0, SIZE, SIZE], [1, 1, 1, 1]]))[:2].T
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    centre = corners.mean(axis=0)
    xy = [tuple(centre + d) for d in ((0, 0), (-12, 0), (12, 0), (0, -12), (0, 12))]
    xy += [tuple(lo + 6), (hi[0] + 20, centre[1])]  # over the fill corner; beyond the grid
    shapes = squares(xy, 6)
    img = he_image(SIZE, SIZE, seed=0, disk=(SIZE / 2, SIZE / 2, SIZE * 0.4))
    sdata = make_sdata(img, image_tf=Affine(rot, input_axes=("x", "y"), output_axes=("x", "y")), shapes=shapes)
    old = sdata.tables["table"]
    var = pd.DataFrame({"gene_ids": list(GENES.values()), "feature_types": "Gene Expression"}, index=list(GENES))
    table = ad.AnnData(X=np.ones((old.n_obs, len(var)), np.int32), obs=old.obs.copy(), var=var)
    del sdata.tables["table"]
    sdata.tables["table"] = TableModel.parse(table, region="cells", region_key="region", instance_key="cell_id")
    path = tmp_path / "rot.zarr"
    sdata.write(path)

    dest = materialize(
        FakeCohort([FakeMember("uidR", path)]),
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids(), TissueMask(method="otsu")],
        role="he",
        roles=ROLES,
        mpp_native={"uidR": 0.25},
    )
    assert validate(dest) == []
    contract = _contract(dest, "dataset_0")
    assert contract.n_outside_image == 2
    assert contract.n_centroid_fallback == 1  # the cell beyond the grid has no pixel
    group = zarr.open_group(dest / "dataset_0", mode="r")
    obs = read_elem(group["obs"])
    assert len(obs) == len(xy)
    flagged = obs.set_index("cell_id")["outside_image"]
    assert flagged.sum() == 2
    assert bool(flagged[shapes.index[5]])
    assert bool(flagged[shapes.index[6]])
    assert not obs.set_index("cell_id")["in_tissue"][shapes.index[5:]].any()
    centres = np.asarray(group["obsm"]["spatial"][...])
    h, w = group["images"]["he"].shape[-2:]
    row = obs["cell_id"].tolist().index(shapes.index[6])
    assert centres[row, 0] >= w  # kept at its true position, beyond the grid

    # the loader pads a crop that leaves the image (spatialbatch reads only the part inside)
    spatialbatch = pytest.importorskip("spatialbatch")
    n = len(xy)  # one batch of every row, so it holds both flagged cells
    loader = spatialbatch.Loader(
        images=["he"], labels=["cells"], crop_px=16, batch_size=n, chunk_size=n, preload_nchunks=1, shuffle=False
    )
    batch = next(iter(loader.use_collection(spatialbatch.SpatialDatasetCollection(dest))))
    he = np.asarray(batch["images"]["he"])
    assert he.shape == (n, 3, 16, 16)
    assert batch["X"].shape[0] == n


def _rasterize(shapes, size):
    """The shapes as a (size, size) uint32 raster with values 1..n (pixel centres inside a shape)."""
    from skimage.draw import polygon

    out = np.zeros((size, size), np.uint32)
    for i, geom in enumerate(shapes.geometry, start=1):
        ring = np.asarray(geom.exterior.coords) - 0.5  # pixel j has its centre at intrinsic j + 0.5
        rr, cc = polygon(ring[:, 1], ring[:, 0], shape=out.shape)
        out[rr, cc] = i
    return out


@pytest.mark.parametrize("runner", ["local", "snakemake"])
def test_labels_cell_without_pixels_takes_its_native_centre_of_mass(tmp_path, runner):
    path = write_member(tmp_path, "uidA", seed=0, labels=True)[0]
    dest = materialize(
        FakeCohort([FakeMember("uidA", path)]),
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25},
        runner=runner,
    )
    assert validate(dest) == []
    assert _contract(dest, "dataset_0").n_centroid_fallback == 1
    group = zarr.open_group(dest / "dataset_0", mode="r")
    obs = read_elem(group["obs"])
    source = sd.read_zarr(path).tables["table"]
    assert len(obs) == source.n_obs
    xy = np.asarray(group["obsm"]["spatial"][...])
    tiny = obs["label_id"].to_numpy() >= obs["label_id"].max() - 1  # the two 1 px cells
    # intrinsic (x + 0.5) at 0.25 um/px is (x + 0.5) / 2 - 0.5 at 0.5 um/px, the pixel-centre convention of Centroids
    expected = source.obsm["spatial"][tiny] / 2 - 0.5
    np.testing.assert_allclose(xy[tiny], expected, atol=0.5)


def test_write_without_spatialbatch_names_the_loader_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "spatialbatch", None)
    with pytest.raises(ImportError, match=r"loader.*spatialdata-db\[loader\]"):
        _materialize.write_collection_dataset()


def test_every_gene_expression_row_is_kept(tmp_path):
    """A row without an Ensembl id and a repeated id are kept, in one member and in the union of two."""
    a = write_member(tmp_path, "uidA", seed=0, dup_id=True)
    b = write_member(tmp_path, "uidB", seed=1, dup_id=True, extra_gene=True)
    cohort = FakeCohort([FakeMember("uidA", a[0]), FakeMember("uidB", b[0])])
    dest = materialize(cohort, tmp_path / "c.zarr", adapters=[Rasterize(None), Centroids()], role="he", roles=ROLES)
    assert validate(dest) == []
    one = GENES["GENE1"]
    union = [one, f"{one}-1", *(GENES[f"GENE{i}"] for i in range(2, 6)), "NOENSEMBL", "NegCtrl_1", "NegCtrl_2"]
    for key, (_, _, counts, var) in zip(("dataset_0", "dataset_1"), (a, b), strict=True):
        table = _table(dest, key)
        assert list(table.var_names) == union
        assert table.var["gene_ids"].tolist()[:2] == [one, one]
        assert pd.isna(table.var.loc["NOENSEMBL", "gene_ids"])
        assert table.var["original_var_name"].tolist()[:3] == ["GENE1", "GENE1B", "GENE2"]  # the symbols are kept
        params = _contract(dest, key).adapters[0].params
        assert (params["no_ensembl_id"], params["duplicates"], params["made_unique"]) == (1, 1, 1)
        assert params["feature_types_from_source"] is True
        assert params["source"] == len(var) == int(table.var["present"].sum())  # no feature dropped
        rows = table.obs["label_id"].to_numpy() - 1
        dense = np.asarray(table.X.todense() if hasattr(table.X, "todense") else table.X)
        for name, source in ((f"{one}-1", "GENE1B"), ("NOENSEMBL", "NOENSEMBL"), (one, "GENE1")):
            np.testing.assert_array_equal(dense[:, union.index(name)], counts[rows, var.index.get_loc(source)])


# -- defaults from the catalog: role, roles, mpp_native --------------------------------------------------------------


@pytest.fixture
def catalog_cohort(tmp_path, monkeypatch):
    """Two members of a technology with default roles, H&E, catalog mpp 0.25, stores recording pixel_size 0.25."""
    monkeypatch.setitem(_defaults.ROLES, ("fake_tech", "he"), ROLES)
    columns = {"technology": "fake_tech", "hne_image": np.True_, "if_image": False, "mpp": 0.25}
    paths = {uid: write_member(tmp_path, uid, seed=i, pixel_size=0.25)[0] for i, uid in enumerate(("uidA", "uidB"))}
    return FakeCohort([FakeMember(uid, path, **columns) for uid, path in paths.items()])


def test_defaults_from_the_catalog(catalog_cohort, tmp_path):
    dest = materialize(catalog_cohort, tmp_path / "c.zarr", adapters=[Rasterize(0.5), Centroids()])
    assert validate(dest) == []
    for key in ("dataset_0", "dataset_1"):
        contract = _contract(dest, key)
        assert (contract.role, contract.grid, contract.mpp_native, contract.mpp_target) == ("he", "target", 0.25, 0.5)
        assert contract.mpp_source == "catalog_checked_store"
    plan, _ = _materialize._prepare(
        catalog_cohort,
        tmp_path / "d.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role=None,
        roles=None,
        mpp_native=None,
        obs_columns=(),
        allow_mixed_grids=False,
    )
    assert (plan["role"], plan["roles"]) == ("he", ROLES)
    assert {uid: m["mpp_native"] for uid, m in plan["members"].items()} == {"uidA": 0.25, "uidB": 0.25}


def test_catalog_mpp_that_disagrees_with_the_store_raises(catalog_cohort, tmp_path):
    catalog_cohort[1].mpp = 0.25 * (1 + 1e-5)
    with pytest.raises(ValueError, match=r"uidB: the catalog's mpp .* differs from the store's own 0.25"):
        materialize(catalog_cohort, tmp_path / "c.zarr", adapters=[Rasterize(0.5), Centroids()])
    # within the tolerance it is taken as is
    catalog_cohort[1].mpp = 0.25 * (1 + 1e-7)
    materialize(catalog_cohort, tmp_path / "d.zarr", adapters=[Rasterize(0.5), Centroids()])
    assert _contract(tmp_path / "d.zarr", "dataset_1").mpp_native == 0.25 * (1 + 1e-7)


def test_explicit_kwargs_override_the_catalog(catalog_cohort, tmp_path):
    catalog_cohort[1].mpp = 0.3  # wrong in the catalog: a given value is not checked against the store
    dest = materialize(
        catalog_cohort, tmp_path / "c.zarr", adapters=[Rasterize(0.5), Centroids()], mpp_native={"uidB": 0.25}
    )
    assert _contract(dest, "dataset_1").mpp_native == 0.25
    assert [_contract(dest, k).mpp_source for k in ("dataset_0", "dataset_1")] == ["catalog_checked_store", "given"]
    # None for a uid means unknown: that member keeps its native grid
    dest = materialize(
        catalog_cohort,
        tmp_path / "d.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        mpp_native={"uidA": None, "uidB": 0.25},
        allow_mixed_grids=True,
    )
    assert [_contract(dest, k).grid for k in ("dataset_0", "dataset_1")] == ["native", "target"]
    assert [_contract(dest, k).mpp_source for k in ("dataset_0", "dataset_1")] == [None, "given"]


def test_store_without_pixel_size_takes_the_catalog_mpp(tmp_path, monkeypatch):
    monkeypatch.setitem(_defaults.ROLES, ("fake_tech", "he"), ROLES)
    path = write_member(tmp_path, "uidA", seed=0)[0]
    member = FakeMember("uidA", path, technology="fake_tech", hne_image=True, mpp=0.25)
    dest = materialize(FakeCohort([member]), tmp_path / "c.zarr", adapters=[Rasterize(0.5), Centroids()])
    assert _contract(dest, "dataset_0").mpp_native == 0.25
    assert _contract(dest, "dataset_0").mpp_source == "catalog_unchecked"  # recorded per dataset, not only logged
    assert validate(dest) == []


def test_null_catalog_mpp_keeps_the_native_grid(catalog_cohort, tmp_path):
    for m in catalog_cohort:
        m.mpp = None
    dest = materialize(catalog_cohort, tmp_path / "c.zarr", adapters=[Rasterize(0.5), Centroids()])
    assert [_contract(dest, k).grid for k in ("dataset_0", "dataset_1")] == ["native", "native"]
    assert [_contract(dest, k).mpp_source for k in ("dataset_0", "dataset_1")] == [None, None]


# -- Visium: the mpp is the array-pitch fit of the store's own spots -----------------------------------------------

VISIUM_MPP = 8.0  # µm per pixel of the synthetic Visium image: one array_col step (50 µm) is 6.25 px
VISIUM_ROLES = {"image": "^images/he$", "table": "^tables/table$"}


def _visium_spots(mpp=VISIUM_MPP, *, jitter=0.0, odd=False, seed=0):
    """Spots on a Visium v1/v2 array: ``array_col`` and ``array_row`` of a spot have the same parity, 100 µm apart."""
    rng = np.random.default_rng(seed)
    step = 50.0 / mpp  # px per array_col step
    cols, rows = zip(*[(c, r) for r in range(7) for c in range(r % 2, 13, 2)], strict=True)
    cols, rows = np.asarray(cols), np.asarray(rows)
    if odd:
        cols = cols + (rows == 3)  # one row off the array's parity
    x = 8 + cols * step + rng.normal(0, jitter, cols.size)
    y = 8 + rows * step * np.sqrt(3) + rng.normal(0, jitter, cols.size)
    circles = gpd.GeoDataFrame(
        {"radius": 27.5 / mpp}, geometry=[shapely.Point(p) for p in zip(x, y, strict=True)], index=np.arange(cols.size)
    )
    return circles, {"array_col": cols, "array_row": rows}


def write_visium_member(tmp_path, uid, *, image_tf=None, **spots):
    """A Visium-like store: H&E image ``he``, spot circles ``cells`` and a table with the spots' array positions."""
    from spatialdata.transformations import Identity

    circles, array = _visium_spots(**spots)
    sdata = make_sdata(he_image(SIZE, SIZE, seed=0), shapes=circles, image_tf=image_tf or Identity(), obs=array)
    var = pd.DataFrame(
        {"gene_ids": list(GENES.values())[:4], "feature_types": ["Gene Expression"] * 4}, index=list(GENES)[:4]
    )
    old = sdata.tables["table"]
    counts = np.random.default_rng(0).poisson(2.0, (old.n_obs, len(var))).astype(np.int32)
    del sdata.tables["table"]
    sdata.tables["table"] = TableModel.parse(
        ad.AnnData(X=counts, obs=old.obs.copy(), var=var), region="cells", region_key="region", instance_key="cell_id"
    )
    path = tmp_path / f"{uid}.zarr"
    sdata.write(path)
    return path


class _LaterCohort(FakeCohort):
    _source = SimpleNamespace(release="2026-12-01")


def test_visium_mpp_is_the_pitch_fit_not_the_stale_catalog(tmp_path):
    # catalog 2026-11-12 and before: 55 / spot_diameter_fullres, 1.18 times too small
    member = FakeMember("uidV", write_visium_member(tmp_path, "uidV"), technology="10x_visium", hne_image=True)
    member.mpp = VISIUM_MPP / 1.1818
    dest = materialize(
        FakeCohort([member]), tmp_path / "c.zarr", adapters=[Rasterize(8.0), Centroids()], roles=VISIUM_ROLES
    )
    contract = _contract(dest, "dataset_0")
    assert contract.mpp_native == pytest.approx(VISIUM_MPP, rel=1e-9)
    assert contract.mpp_source == "store_pitch_fit"
    assert validate(dest) == []


def test_visium_catalog_mpp_off_the_fit_raises_after_the_stale_releases(tmp_path):
    path = write_visium_member(tmp_path, "uidV")
    member = FakeMember("uidV", path, technology="10x_visium", hne_image=True, mpp=VISIUM_MPP / 1.1818)
    with pytest.raises(ValueError, match=r"uidV: the catalog's mpp .* differs from the pitch fit .* by more than 0.02"):
        materialize(
            _LaterCohort([member]), tmp_path / "c.zarr", adapters=[Rasterize(8.0), Centroids()], roles=VISIUM_ROLES
        )
    member.mpp = VISIUM_MPP * 1.01  # within PITCH_RTOL: the fit is still the value
    dest = materialize(
        _LaterCohort([member]), tmp_path / "d.zarr", adapters=[Rasterize(8.0), Centroids()], roles=VISIUM_ROLES
    )
    assert _contract(dest, "dataset_0").mpp_native == pytest.approx(VISIUM_MPP, rel=1e-9)


def test_pitch_fit_on_a_downscaled_image():
    from spatialdata.transformations import Scale

    # a hires-like image: one image pixel is 2 units of the spots' frame
    circles, array = _visium_spots()
    sdata = make_sdata(
        he_image(SIZE // 2, SIZE // 2), shapes=circles, image_tf=Scale([2.0, 2.0], axes=("x", "y")), obs=array
    )
    assert _defaults.pitch_mpp(sdata, "he") == pytest.approx(2 * VISIUM_MPP, rel=1e-9)
    # the circles' radius does not enter: spatialdata-io wrote spot_diameter_fullres / 2 before
    sdata.shapes["cells"]["radius"] = 50.0
    assert _defaults.pitch_mpp(sdata, "he") == pytest.approx(2 * VISIUM_MPP, rel=1e-9)


@pytest.mark.parametrize(
    ("spots", "match"),
    [({"odd": True}, "not on a Visium v1/v2 array"), ({"jitter": 2.0}, r"median .* µm, more than 5 µm")],
)
def test_pitch_fit_refuses_spots_off_the_array(spots, match):
    circles, array = _visium_spots(**spots)
    sdata = make_sdata(he_image(SIZE, SIZE), shapes=circles, obs=array)
    with pytest.raises(ValueError, match=match):
        _defaults.pitch_mpp(sdata, "he")


def test_visium_without_array_positions_never_takes_the_stale_catalog(caplog):
    circles, _ = _visium_spots()
    sdata = make_sdata(he_image(SIZE, SIZE), shapes=circles)
    member = SimpleNamespace(uid="uidV", technology="10x_visium", mpp=VISIUM_MPP / 1.1818)
    assert _defaults.resolve_mpp(member, sdata, "he", given={}, release="2026-11-12") == (None, None)
    assert "known stale" in caplog.text
    assert _defaults.resolve_mpp(member, sdata, "he", given={}, release="2026-12-01") == (
        VISIUM_MPP / 1.1818,
        "catalog_unchecked",
    )
    assert _defaults.resolve_mpp(member, sdata, "he", given={"uidV": 7.0}, release="2026-11-12") == (7.0, "given")


@pytest.mark.parametrize(
    ("technology", "release", "stale"),
    [
        ("10x_visium", "2026-11-12", True),
        ("10x_visium", "2026-10-01", True),
        ("10x_visium", "2026-11-13", False),
        ("10x_visium", "unknown", True),
        ("10x_visium", None, True),
        ("10x_visium_hd", "2026-11-12", False),
        ("10x_atera", "2026-11-12", False),
    ],
)
def test_catalog_is_stale(technology, release, stale):
    assert _defaults.catalog_is_stale(technology, release) is stale


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"hne_image": None, "if_image": None}, "pass role"),
        ({"technology": "other_tech"}, "pass roles: the members have several technologies"),
    ],
)
def test_defaults_that_cannot_be_inferred_ask_for_the_kwarg(catalog_cohort, tmp_path, change, match):
    catalog_cohort[0].__dict__.update(change)
    with pytest.raises(ValueError, match=match):
        materialize(catalog_cohort, tmp_path / "c.zarr", adapters=[Rasterize(0.5), Centroids()])


def test_technology_without_default_roles_asks_for_roles(catalog_cohort, tmp_path):
    for m in catalog_cohort:
        m.technology = "other_tech"
    with pytest.raises(ValueError, match="pass roles: no default roles for technology 'other_tech'"):
        materialize(catalog_cohort, tmp_path / "c.zarr", adapters=[Rasterize(0.5), Centroids()])
    # explicit roles stand in
    dest = materialize(catalog_cohort, tmp_path / "d.zarr", adapters=[Rasterize(0.5), Centroids()], roles=ROLES)
    assert validate(dest) == []


def test_atera_defaults_match_the_reader():
    assert _defaults.ROLES[("10x_atera", "he")] == {
        "image": "^images/he_image$",
        "table": "^tables/table$",
        "labels": "^labels/cell_labels$",
    }


# -- totals: obs["total_counts"], var["mean_counts"] and var["var_counts"] -------------------------------------------


@pytest.mark.parametrize("lazy", [True, False], ids=["dask-X", "in-memory-X"])
def test_totals_are_written_and_validate(cohort, tmp_path, monkeypatch, lazy):
    members, _ = cohort
    if not lazy:  # members opened whole (to_sdata), as a remote member is
        for m in members:
            monkeypatch.setattr(m, "_local_store", None)
    dest = materialize(
        members,
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25, "uidB": 0.25},
    )
    assert validate(dest) == []
    for key in ("dataset_0", "dataset_1"):
        table = _table(dest, key)
        X = np.asarray(table.X.todense() if hasattr(table.X, "todense") else table.X, dtype=np.float64)
        ge = (table.var["feature_types"] == "Gene Expression").to_numpy()
        np.testing.assert_array_equal(table.obs["total_counts"].to_numpy(), X[:, ge].sum(axis=1))
        np.testing.assert_allclose(table.var["mean_counts"].to_numpy(), X.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(table.var["var_counts"].to_numpy(), X.var(axis=0, ddof=1), rtol=1e-12, atol=1e-12)
        filled = ~table.var["present"].to_numpy()
        assert (table.var.loc[filled, ["mean_counts", "var_counts"]].to_numpy() == 0).all()


def test_source_total_counts_is_kept_under_a_new_name(tmp_path):
    path, *_ = write_member(tmp_path, "uidA", seed=0)
    sdata = sd.read_zarr(path)
    sdata.tables["table"].obs["total_counts"] = 7.0
    path2 = tmp_path / "with_totals.zarr"
    sdata.write(path2)
    dest = materialize(
        FakeCohort([FakeMember("uidA", path2)]),
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25},
    )
    obs = _table(dest, "dataset_0").obs
    assert (obs["total_counts_source"] == 7.0).all()
    assert validate(dest) == []


def test_mpp_source_needs_an_mpp():
    from sddb.contract import DatasetContract

    base = _contract_dict()
    assert DatasetContract.model_validate({**base, "mpp_source": "catalog_unchecked"}).mpp_source == "catalog_unchecked"
    with pytest.raises(ValueError, match="mpp_source is set but mpp_native is unknown"):
        DatasetContract.model_validate({**base, "grid": "native", "mpp_native": None, "mpp_source": "given"})
    with pytest.raises(ValueError, match="mpp_source"):
        DatasetContract.model_validate({**base, "mpp_source": "guessed"})


def _contract_dict():
    return {
        "version": "0.1.0",
        "uid": "u",
        "study_id": "s",
        "catalog_release": "r",
        "role": "he",
        "grid": "target",
        "mpp_target": 0.5,
        "mpp_native": 0.25,
        "upsampled": False,
        "allow_upsample": False,
        "allow_mixed_grids": False,
        "adapters": [],
    }


# -- stores: a local or derived store under a catalog uid; only the role elements fetched --------------------------


def test_stores_open_a_local_store_under_the_catalog_record(tmp_path):
    path = write_member(tmp_path, "local", seed=0)[0]
    # the catalog's store is not there: the member is opened from `stores` and keeps its record and the release
    member = FakeMember("uidA", tmp_path / "missing.zarr")
    dest = materialize(
        FakeCohort([member]),
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25},
        stores={"uidA": path},
    )
    contract = _contract(dest, "dataset_0")
    assert (contract.uid, contract.catalog_release) == ("uidA", "2026-11-01")
    assert validate(dest) == []
    with pytest.raises(ValueError, match=r"stores has \['uidZ'\], which are not members"):
        materialize(
            FakeCohort([member]),
            tmp_path / "d.zarr",
            adapters=[Rasterize(0.5), Centroids()],
            role="he",
            roles=ROLES,
            mpp_native={"uidA": 0.25},
            stores={"uidZ": path},
        )


class _RemoteMember(FakeMember):
    """A member with a remote store: it lists its elements, and fetches only those asked for."""

    zarr_url = "s3://bucket/uidA.zarr"

    def resolve(self, roles):
        return {role: {"image": "images/he", "table": "tables/table"}[role] for role in roles}

    def _local_store(self, elements=None):
        self.fetched = elements
        return self._path


def test_remote_member_fetches_only_the_role_elements(tmp_path):
    member = _RemoteMember("uidA", write_member(tmp_path, "uidA", seed=0)[0])
    plan, _ = _materialize._prepare(
        FakeCohort([member]),
        tmp_path / "c.zarr",
        adapters=[Rasterize(0.5), Centroids()],
        role="he",
        roles=ROLES,
        mpp_native={"uidA": 0.25},
        obs_columns=(),
        allow_mixed_grids=False,
    )
    assert member.fetched == ["images/he", "tables/table"]
    assert plan["members"]["uidA"]["path"] == str(member._path.absolute())
