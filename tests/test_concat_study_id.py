import anndata as ad
import numpy as np
import pandas as pd

from sddb import concat


def _fake_group(adatas):
    """Patch concat so each member returns a prebuilt AnnData instead of reading zarr."""
    it = iter(adatas)

    def read_zarr(_group):
        return next(it)

    return read_zarr


def test_to_anndata_adds_uid_and_study_id(monkeypatch):
    a = ad.AnnData(np.ones((2, 2)))
    b = ad.AnnData(np.ones((3, 2)))
    monkeypatch.setattr(concat.ad, "read_zarr", _fake_group([a, b]))
    monkeypatch.setattr(concat, "_open_group", lambda url: url)
    members = pd.DataFrame(
        {"uid": ["u1", "u2"], "zarr_url": ["s3://b/1.zarr", "s3://b/2.zarr"], "study_id": ["s_a", "s_b"]}
    )
    out = concat.concat_tables(members)
    assert "uid" in out.obs
    assert "study_id" in out.obs
    assert set(out.obs["study_id"].dropna()) == {"s_a", "s_b"}


def test_to_anndata_without_study_id_column(monkeypatch):
    a = ad.AnnData(np.ones((2, 2)))
    monkeypatch.setattr(concat.ad, "read_zarr", _fake_group([a]))
    monkeypatch.setattr(concat, "_open_group", lambda url: url)
    members = pd.DataFrame({"uid": ["u1"], "zarr_url": ["s3://b/1.zarr"]})
    out = concat.concat_tables(members)  # must not raise
    assert "study_id" in out.obs
    assert out.obs["study_id"].isna().all()


def test_verify_param_removed():
    import inspect

    assert "verify" not in inspect.signature(concat.concat_tables).parameters
