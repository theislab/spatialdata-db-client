from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData

from sddb.concat import concat_tables


def _store(path, n):
    SpatialData(tables={"table": ad.AnnData(X=np.ones((n, 3), "float32"))}).write(path)
    return str(path)


def test_concat_reads_only_tables(tmp_path):
    a = _store(tmp_path / "a.zarr", 4)
    b = _store(tmp_path / "b.zarr", 6)
    members = pd.DataFrame(
        {"uid": ["a", "b"], "zarr_url": [a, b], "fingerprint": ["x", "y"], "default_table": ["table", "table"]}
    )
    adata = concat_tables(members)
    assert adata.n_obs == 10
    assert set(adata.obs["uid"]) == {"a", "b"}


def test_concat_missing_table_names_member(tmp_path):
    a = _store(tmp_path / "a.zarr", 4)
    members = pd.DataFrame({"uid": ["a"], "zarr_url": [a], "fingerprint": ["x"], "default_table": ["nope"]})
    with pytest.raises(KeyError) as ei:
        concat_tables(members)
    assert "'a'" in str(ei.value)


def test_concat_without_default_table_column(tmp_path):
    a = _store(tmp_path / "a.zarr", 2)
    members = pd.DataFrame({"uid": ["a"], "zarr_url": [a]})
    assert concat_tables(members).n_obs == 2


def test_cohort_delegators(tmp_path):
    from sddb.cohort import SpatialDataCohort

    a = _store(tmp_path / "a.zarr", 3)
    df = pd.DataFrame({"uid": ["a"], "zarr_url": [a], "fingerprint": ["x"]})
    assert SpatialDataCohort(df).to_anndata().n_obs == 3


@pytest.mark.network
def test_concat_remote_catalog_objects():
    from sddb import Catalog

    cohort = Catalog().query()
    df = cohort.to_df().head(2)
    assert len(df) == 2
    adata = cohort[:2].to_anndata()
    assert adata.n_obs > 0
    assert set(adata.obs["uid"]) == set(df["uid"].astype(str))


def _members(tmp_path, sizes):
    urls = [_store(tmp_path / f"{i}.zarr", n) for i, n in enumerate(sizes)]
    return pd.DataFrame(
        {
            "uid": [f"u{i}" for i in range(len(sizes))],
            "zarr_url": urls,
            "organism": ["human", "mouse", "human"][: len(sizes)],
        }
    )


def test_concat_obs_columns_follow_members_under_parallel_reads(tmp_path):
    members = _members(tmp_path, [2, 3, 4])
    adata = concat_tables(members, obs_columns=["organism"], max_workers=3)
    by_uid = adata.obs.groupby("uid", observed=True)["organism"].first().to_dict()
    assert by_uid == {"u0": "human", "u1": "mouse", "u2": "human"}
    assert adata.obs["uid"].value_counts().to_dict() == {"u2": 4, "u1": 3, "u0": 2}
    assert adata.obs["organism"].dtype == "category"


def test_concat_unknown_obs_column_raises(tmp_path):
    with pytest.raises(KeyError, match="tissue"):
        concat_tables(_members(tmp_path, [2]), obs_columns=["tissue"])


def test_concat_elements_skips_layers(tmp_path):
    path = tmp_path / "l.zarr"
    a = ad.AnnData(X=np.ones((2, 3), "float32"), layers={"counts": np.ones((2, 3), "float32")})
    SpatialData(tables={"table": a}).write(path)
    members = pd.DataFrame({"uid": ["l"], "zarr_url": [str(path)]})
    assert "counts" in concat_tables(members).layers
    assert "counts" not in concat_tables(members, elements=("X", "obs", "var")).layers


class _MouseOnly:
    """Skips non-mouse rows cheaply, tables without var names authoritatively, keeps one gene."""

    name = "mouse-only"
    version = "1"

    def __init__(self, keep=True):
        self.keep = keep

    def requirements(self):
        return {"elements": ("X", "obs", "var")}

    def validate_meta(self, row):
        return [] if row["organism"] == "mouse" else [f"organism is {row['organism']}"]

    def validate(self, adata, row):
        return [] if adata.n_vars else ["no genes"]

    def build(self, adata, row):
        return adata[:, :1].copy() if self.keep else None


def test_concat_adapter_maps_and_reports_skips(tmp_path):
    members = _members(tmp_path, [2, 3, 4])
    members.loc[0, "zarr_url"] = str(tmp_path / "missing.zarr")  # skipped by validate_meta, so never read
    adata = concat_tables(members, adapter=_MouseOnly())
    assert adata.shape == (3, 1)
    assert set(adata.obs["uid"]) == {"u1"}
    assert adata.uns["sddb"] == {
        "adapter": "mouse-only",
        "adapter_version": "1",
        "dropped": {"u0": "organism is human", "u2": "organism is human"},
    }


def test_concat_adapter_build_none_is_reported(tmp_path):
    members = _members(tmp_path, [2, 3])
    members["organism"] = "mouse"
    with pytest.raises(ValueError, match="dropped by mouse-only.build"):
        concat_tables(members, adapter=_MouseOnly(keep=False))


def test_concat_adapter_requirements_select_elements(tmp_path):
    path = tmp_path / "l.zarr"
    a = ad.AnnData(X=np.ones((2, 3), "float32"), layers={"counts": np.ones((2, 3), "float32")})
    SpatialData(tables={"table": a}).write(path)
    members = pd.DataFrame({"uid": ["l"], "zarr_url": [str(path)], "organism": ["mouse"]})
    assert "counts" not in concat_tables(members, adapter=_MouseOnly()).layers


def test_table_adapter_protocol():
    from sddb import TableAdapter

    assert isinstance(_MouseOnly(), TableAdapter)


# gene_key: a 10x-style member (symbols in var_names, Ensembl in gene_ids, one control probe) and a
# HEST-style member (versioned Ensembl var_names, symbols in var["symbol"], junk gene_id_raw)
_TENX_VAR = pd.DataFrame(
    {"gene_ids": ["ENSG00000186092", "ENSG00000187634", "ENSG00000188976", "NegControlProbe_1"]},
    index=["OR4F5", "SAMD11", "NOC2L", "NegControlProbe_1"],
)
_HEST_VAR = pd.DataFrame(
    {"symbol": ["OR4F5", "SAMD11", "NOC2L", "KLHL17"], "gene_id_raw": ["Unspecified_gene_id"] * 4},
    index=["ENSG00000186092.5", "ENSG00000187634", "ENSG00000188976", "ENSG00000187961"],
)


def _var_store(path, var, n=2):
    SpatialData(tables={"table": ad.AnnData(X=np.ones((n, len(var)), "float32"), var=var)}).write(path)
    return str(path)


def _mixed(tmp_path):
    return pd.DataFrame(
        {
            "uid": ["tenx", "hest"],
            "zarr_url": [_var_store(tmp_path / "t.zarr", _TENX_VAR), _var_store(tmp_path / "h.zarr", _HEST_VAR)],
        }
    )


def _ours(record):
    return [str(w.message) for w in record if "gene_key" in str(w.message) or "members share" in str(w.message)]


def test_gene_key_none_keeps_var_names_and_warns_low_overlap(tmp_path):
    with pytest.warns(UserWarning, match="members share") as record:
        adata = concat_tables(_mixed(tmp_path))
    assert set(adata.var_names) == {*_TENX_VAR.index, *_HEST_VAR.index}
    assert adata.var.columns.empty  # unchanged: no var merge without gene_key
    (msg,) = _ours(record)
    assert msg == (
        "members share 0 genes, 0% of the smallest member's 4; "
        "pass gene_key='ensembl' or 'symbol' to align gene ids across sources"
    )


def test_gene_key_none_makes_repeated_var_names_unique(tmp_path):
    var = pd.DataFrame(index=["ENSG00000186092", "ENSG00000187634", "ENSG00000186092"])
    members = pd.DataFrame(
        {
            "uid": ["rep", "tenx"],
            "zarr_url": [_var_store(tmp_path / "r.zarr", var), _var_store(tmp_path / "t.zarr", _TENX_VAR)],
        }
    )
    with pytest.warns(UserWarning, match="repeated var_names") as record:
        adata = concat_tables(members)
    assert {"ENSG00000186092", "ENSG00000186092-1", "ENSG00000187634"} <= set(adata.var_names)
    assert set(_TENX_VAR.index) <= set(adata.var_names)  # members without repeats are unchanged
    (msg,) = [str(w.message) for w in record if "repeated" in str(w.message)]
    assert msg == (
        "made 1 repeated var_names unique with a '-1' style suffix in members ['rep']; "
        "pass gene_key='ensembl' or 'symbol' to merge them instead"
    )


def test_gene_key_ensembl_aligns_members(tmp_path):
    with pytest.warns(UserWarning, match="gene_key='ensembl'") as record:
        adata = concat_tables(_mixed(tmp_path), gene_key="ensembl")
    assert sorted(adata.var_names) == ["ENSG00000186092", "ENSG00000187634", "ENSG00000187961", "ENSG00000188976"]
    assert adata.var.loc["ENSG00000186092", "original_var_name"] == "OR4F5"  # first member wins
    assert adata.var.loc["ENSG00000187961", "original_var_name"] == "ENSG00000187961"
    (msg,) = _ours(record)  # the control probe is dropped; the overlap is 3 of 3, so no overlap warning
    assert msg == "gene_key='ensembl': dropped 1 features without an Ensembl id and 0 duplicates across 1 members"
    shared = adata[:, ["ENSG00000186092", "ENSG00000187634", "ENSG00000188976"]].X
    assert not np.isnan(np.asarray(shared, dtype=float)).any()  # both members fill the shared genes


def test_gene_key_symbol_aligns_members(tmp_path):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        adata = concat_tables(_mixed(tmp_path), gene_key="symbol")
    assert sorted(adata.var_names) == ["KLHL17", "NOC2L", "NegControlProbe_1", "OR4F5", "SAMD11"]
    assert adata.var.loc["KLHL17", "original_var_name"] == "ENSG00000187961"
    assert _ours(record) == []


def test_gene_key_duplicates_keep_first(tmp_path):
    var = pd.DataFrame({"symbol": ["A", "B", "A", ""]}, index=[f"ENSG0000000000{i}" for i in range(4)])
    members = pd.DataFrame({"uid": ["d"], "zarr_url": [_var_store(tmp_path / "d.zarr", var)]})
    with pytest.warns(UserWarning, match="dropped 1 features without a symbol and 1 duplicates across 1 members"):
        adata = concat_tables(members, gene_key="symbol")
    assert list(adata.var_names) == ["A", "B"]
    assert list(adata.var["original_var_name"]) == ["ENSG00000000000", "ENSG00000000001"]


def test_gene_key_without_usable_source_names_member(tmp_path):
    var = pd.DataFrame({"note": ["x", "y"]}, index=["OR4F5", "SAMD11"])
    members = pd.DataFrame({"uid": ["nokey"], "zarr_url": [_var_store(tmp_path / "n.zarr", var)]})
    with pytest.raises(ValueError, match=r"'nokey'.*gene_key='ensembl'.*\['note'\]"):
        concat_tables(members, gene_key="ensembl")


def test_gene_key_rejects_unknown_key(tmp_path):
    with pytest.raises(ValueError, match="gene_key must be"):
        concat_tables(_mixed(tmp_path), gene_key="entrez")


def test_gene_key_adds_var_to_elements(tmp_path):
    with pytest.warns(UserWarning, match="dropped 1 features"):
        adata = concat_tables(_mixed(tmp_path), elements=("X", "obs"), gene_key="ensembl")
    assert adata.n_vars == 4
    assert all(v.startswith("ENSG") for v in adata.var_names)


def test_cohort_to_anndata_passes_gene_key(tmp_path):
    from sddb.cohort import SpatialDataCohort

    with pytest.warns(UserWarning, match="dropped 1 features"):
        assert SpatialDataCohort(_mixed(tmp_path)).to_anndata(gene_key="ensembl").n_vars == 4


@pytest.mark.network
def test_gene_key_real_10x_and_hest_pair():
    """The failure case from the paper: a GEO 10x and a HEST Visium sample share no var_names as stored."""
    from sddb import Catalog
    from sddb.concat import _read_table, _table_name

    cohort = Catalog().cohort(uids=["qEP9pd4LlyKeufe20000", "USx288NLowiqKbV30000"])
    # as stored (gene_key=None); the HEST member repeats 2 var_names, which to_anndata makes unique
    stored = [
        set(_read_table(f"{str(r['zarr_url']).rstrip('/')}/tables/{_table_name(r)}", ("var",)).var_names)
        for _, r in cohort.members.iterrows()
    ]
    shared = {None: len(stored[0] & stored[1])}
    with pytest.warns(UserWarning, match=r"made 2 repeated var_names unique .*'USx288NLowiqKbV30000'"):
        as_stored = cohort.to_anndata(elements=("X", "var", "obs"))  # used to raise InvalidIndexError
    assert as_stored.n_vars == 19850 + 17943  # nothing shared, the 2 repeats kept as "-1" copies
    for key in ("ensembl", "symbol"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            each = [cohort[i : i + 1].to_anndata(elements=("X", "var", "obs"), gene_key=key) for i in range(2)]
            both = cohort.to_anndata(elements=("X", "var", "obs"), gene_key=key)
        shared[key] = len(set(each[0].var_names) & set(each[1].var_names))
        assert both.n_vars == sum(a.n_vars for a in each) - shared[key]  # outer join
        print(f"{key}: per-member n_vars {[a.n_vars for a in each]}, joined {both.n_vars}")
    print(f"stored n_vars {[len(v) for v in stored]}; shared genes: {shared}")
    assert shared[None] == 0
    assert shared["ensembl"] > 0
    assert shared["symbol"] > 0


# spatial: tables that annotate a shapes region, with instance ids in a different order than the shapes
_XY = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])


def _shapes(kind):
    import geopandas as gpd
    from shapely.geometry import box
    from spatialdata.models import ShapesModel

    ids = pd.Index(["c0", "c1", "c2"])
    if kind == "circles":
        gdf = gpd.GeoDataFrame({"radius": [1.0] * 3}, geometry=gpd.points_from_xy(*_XY.T), index=ids)
    else:  # squares of side 2 centred on _XY
        gdf = gpd.GeoDataFrame(geometry=[box(x - 1, y - 1, x + 1, y + 1) for x, y in _XY], index=ids)
    return ShapesModel.parse(gdf)


def _region_store(path, kind="circles", obsm=None, region="cells"):
    from spatialdata.models import TableModel

    obs = pd.DataFrame({"cell_id": ["c2", "c0", "c1"], "region": pd.Categorical(["cells"] * 3)})
    obs.index = [f"row{i}" for i in range(3)]
    table = TableModel.parse(
        ad.AnnData(X=np.ones((3, 2), "float32"), obs=obs, obsm=obsm or {}),
        region="cells",
        region_key="region",
        instance_key="cell_id",
    )
    shapes = {region: _shapes(kind)} if region else {}
    SpatialData(shapes=shapes, tables={"table": table}).write(path)
    return str(path)


def _one(path, uid="m"):
    return pd.DataFrame({"uid": [uid], "zarr_url": [path]})


@pytest.mark.parametrize("kind", ["circles", "polygons"])
def test_spatial_auto_uses_centroids_matched_by_instance_key(tmp_path, kind):
    adata = concat_tables(_one(_region_store(tmp_path / "s.zarr", kind)))
    np.testing.assert_allclose(adata.obsm["spatial"], _XY[[2, 0, 1]])  # obs order c2, c0, c1


def test_spatial_centroids_without_obsm_element(tmp_path):
    adata = concat_tables(_one(_region_store(tmp_path / "s.zarr")), elements=("X", "var"), spatial="centroids")
    np.testing.assert_allclose(adata.obsm["spatial"], _XY[[2, 0, 1]])


def test_spatial_auto_keeps_stored_obsm(tmp_path):
    stored = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 0.0]])
    path = _region_store(tmp_path / "s.zarr", obsm={"spatial": stored})
    np.testing.assert_allclose(concat_tables(_one(path)).obsm["spatial"], stored)
    np.testing.assert_allclose(concat_tables(_one(path), spatial="obsm").obsm["spatial"], stored)
    np.testing.assert_allclose(concat_tables(_one(path), spatial="centroids").obsm["spatial"], _XY[[2, 0, 1]])


def test_spatial_auto_replaces_stored_obsm_with_nan_rows(tmp_path):
    stored = np.array([[5.0, 6.0], [np.nan, np.nan], [9.0, 0.0]])
    path = _region_store(tmp_path / "s.zarr", obsm={"spatial": stored})
    np.testing.assert_allclose(concat_tables(_one(path)).obsm["spatial"], _XY[[2, 0, 1]])


def test_spatial_obsm_leaves_members_without_coordinates(tmp_path):
    members = pd.concat([_one(_region_store(tmp_path / "s.zarr"), "s"), _one(_store(tmp_path / "t.zarr", 2), "t")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adata = concat_tables(members, spatial="obsm")
    assert "spatial" not in adata.obsm


def test_spatial_warns_once_for_members_without_source(tmp_path):
    members = pd.concat(
        [
            _one(_region_store(tmp_path / "s.zarr"), "s"),
            _one(_store(tmp_path / "t.zarr", 2), "t"),  # no region at all
            _one(_region_store(tmp_path / "n.zarr", region=None), "n"),  # region element absent
        ]
    )
    with pytest.warns(UserWarning, match="spatial='auto'") as record:
        adata = concat_tables(members)
    (msg,) = [str(w.message) for w in record if "spatial=" in str(w.message)]
    assert msg == (
        "spatial='auto': no usable coordinates for 2 members: "
        "'n' (no obsm['spatial'] and region 'cells' is not in the store); "
        "'t' (no obsm['spatial'] and the table annotates no region)"
    )
    xy = adata.obsm["spatial"]
    by_uid = adata.obs["uid"].astype(str).to_numpy()
    np.testing.assert_allclose(xy[by_uid == "s"], _XY[[2, 0, 1]])
    assert np.isnan(xy[by_uid != "s"]).all()


def test_spatial_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="spatial must be"):
        concat_tables(_one(_store(tmp_path / "t.zarr", 2)), spatial="xy")


def test_cohort_to_anndata_passes_spatial(tmp_path):
    from sddb.cohort import SpatialDataCohort

    path = _region_store(tmp_path / "s.zarr", obsm={"spatial": np.zeros((3, 2))})
    adata = SpatialDataCohort(_one(path)).to_anndata(spatial="centroids")
    np.testing.assert_allclose(adata.obsm["spatial"], _XY[[2, 0, 1]])


@pytest.mark.network
@pytest.mark.parametrize(
    ("uid", "n_obs"),
    [("rdngUwLr6XlVEmpD0000", 16324), ("r3rkbHVtKnv9hVKq0000", 5505), ("wfVFQEPZXA65gc8S0000", 48944)],
)
def test_spatial_real_tables_get_centroids(uid, n_obs):
    """Allen MERSCOPE/MERFISH (cell_circles) and Bruker CosMx (cells_polygons) tables store no obsm["spatial"]."""
    from sddb import Catalog

    cohort = Catalog().cohort(uids=[uid])
    assert "spatial" not in cohort.to_anndata(elements=("obs", "obsm"), spatial="obsm").obsm
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        xy = cohort.to_anndata(elements=("obs",)).obsm["spatial"]
    assert xy.shape == (n_obs, 2)
    assert not np.isnan(xy).any()
    print(
        f"{uid}: {len(xy)} rows with coordinates, x {xy[:, 0].min():.1f}..{xy[:, 0].max():.1f}, "
        f"y {xy[:, 1].min():.1f}..{xy[:, 1].max():.1f}"
    )


def test_spatial_labels_region_counts_as_no_source(tmp_path):
    from spatialdata.models import Labels2DModel, TableModel

    obs = pd.DataFrame({"cell_id": [1, 2], "region": pd.Categorical(["seg"] * 2)}, index=["a", "b"])
    table = TableModel.parse(ad.AnnData(X=np.ones((2, 1), "float32"), obs=obs), "seg", "region", "cell_id")
    labels = Labels2DModel.parse(np.array([[0, 1], [2, 2]], dtype="uint8"), dims=("y", "x"))
    SpatialData(labels={"seg": labels}, tables={"table": table}).write(tmp_path / "l.zarr")
    with pytest.warns(UserWarning, match=r"'l' \(no obsm\['spatial'\] and region 'seg' is a labels element\)"):
        adata = concat_tables(_one(str(tmp_path / "l.zarr"), "l"))
    assert "spatial" not in adata.obsm
