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
    # as stored (gene_key=None); read var directly, since this HEST member repeats 2 var_names and
    # ad.concat refuses a non-unique var index
    stored = [
        set(_read_table(f"{str(r['zarr_url']).rstrip('/')}/tables/{_table_name(r)}", ("var",)).var_names)
        for _, r in cohort.members.iterrows()
    ]
    shared = {None: len(stored[0] & stored[1])}
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
