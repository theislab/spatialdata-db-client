"""``materialize`` (Task 4 of T144): a synthetic cohort written as a loader collection, rerun and read back."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import spatialdata as sd
import zarr
from anndata.io import read_elem
from scipy import ndimage
from spatialdata.models import TableModel
from tests.test_loader_adapters import he_image, make_sdata, squares

from sddb.contract import DatasetContract, validate
from sddb.loader import Centroids, Rasterize, StainNormalize, TissueMask, _materialize, default_chain, materialize

ROLES = {"image": "^images/he$", "table": "^tables/table$"}
SIZE = 96
# Gene Expression features with an Ensembl id; uidB also measures GENE5
GENES = {f"GENE{i}": f"ENSG{i:011d}" for i in range(1, 6)}


class FakeMember:
    """The part of :class:`sddb.Dataset` that ``materialize`` uses: uid, study_id, catalog columns, ``to_sdata``."""

    def __init__(self, uid, path, tissue="breast"):
        self.uid, self.study_id, self.tissue, self._path = uid, f"study_{uid}", tissue, path

    def to_sdata(self):
        return sd.read_zarr(self._path)


class FakeCohort(list):
    _source = SimpleNamespace(release="2026-11-01")


def write_member(tmp_path, uid, *, seed, extra_gene=False, tiny=False, labels=False):
    """A member store: H&E tissue disc, square cells, a table keyed by symbol with Ensembl ids and controls."""
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

    shared = [GENES[f"GENE{i}"] for i in range(1, 5)]  # GENE5 only in uidB; the controls and NOENSEMBL left out
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
        assert genes["non_gene_expression"] == 2
        assert genes["no_ensembl_id"] == 1
        assert genes["not_shared"] == (1 if uid == "uidB" else 0)
        references.append(contract.stain_reference)

        table = _table(dest, key)
        assert list(table.var_names) == shared
        assert set(table.obs["uid"]) == {uid}
        assert set(table.obs["tissue"]) == {"breast"}
        _, _, counts, var = sources[uid]
        # X holds the source counts of the shared genes, row for row by label id
        source_rows = table.obs["label_id"].to_numpy() - 1
        cols = [var.index.get_loc(g) for g in var.index[var["gene_ids"].isin(shared)]]
        np.testing.assert_array_equal(
            np.asarray(table.X.todense() if hasattr(table.X, "todense") else table.X), counts[source_rows][:, cols]
        )
        assert (state / "done" / uid).is_file()
        assert json.loads((state / "done" / uid).read_text())["key"] == key
    assert references[0] is not None
    assert references[0] == references[1]  # one pooled reference for the cohort
    assert "2 features that are not Gene Expression" in caplog.text
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
def test_labels_cell_without_pixels_takes_its_table_centroid(tmp_path, runner):
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
