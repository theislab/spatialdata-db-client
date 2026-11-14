"""Tests for the sddb loader contract: models and validator, on synthetic collections with one defect each."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

# the loader extra
pytest.importorskip("pydantic")
import zarr
from scipy import ndimage
from tests._contract_fixture import N_VARS, VAR_NAMES, base_contract, draw_cells, write_collection

from sddb.contract import ContractError, DatasetContract, assert_valid, validate, validate_dataset


def _issues(tmp_path, *args, **kwargs) -> list[str]:
    return validate(write_collection(tmp_path / "c.zarr", *args, **kwargs))


def test_conforming_fixture_has_no_issues(tmp_path):
    path = write_collection(tmp_path / "c.zarr")
    assert validate(path) == []
    assert validate_dataset(zarr.open_group(path, mode="r")["dataset_0"]) == []
    assert_valid(path)


def test_conforming_if_fixture_has_no_issues(tmp_path):
    assert _issues(tmp_path, role="if") == []


def test_wrong_dtype_for_role(tmp_path):
    issues = _issues(tmp_path, overrides={0: {"image_dtype": np.uint16}})
    assert len(issues) == 1
    assert issues[0].startswith("dataset_0:")
    assert "uint8" in issues[0]
    with pytest.raises(ContractError) as e:
        assert_valid(tmp_path / "c.zarr")
    assert e.value.issues == issues


def test_labels_shape_mismatch(tmp_path):
    issues = _issues(tmp_path, overrides={1: {"labels_shape": (100, 140)}})
    assert len(issues) == 1
    assert issues[0].startswith("dataset_1:")
    assert "labels/cells" in issues[0]
    assert "(100, 140)" in issues[0]


def test_centroid_outside_image(tmp_path):
    issues = _issues(tmp_path, overrides={0: {"move_centroid": (140.0, 10.0)}})
    assert len(issues) == 1
    assert "obsm['spatial']" in issues[0]
    assert "outside" in issues[0]


def test_missing_label_instance(tmp_path):
    # 1 erased cell in 200 is within the 99% centroid tolerance, so only the hard check fails.
    issues = _issues(tmp_path, height=200, width=280, n_cells=200, overrides={0: {"erase_cells": 1}})
    assert len(issues) == 1
    assert "no pixel" in issues[0]
    assert "label_id [200]" in issues[0]


@pytest.mark.parametrize(
    ("label_ids", "match"),
    [
        ([0] * 60, "<= 0"),
        ([1, *range(1, 60)], "not unique"),
        ([-1, *range(2, 61)], "<= 0"),
        ([2**40, *range(2, 61)], "max of labels/cells"),
    ],
    ids=["all-zero", "duplicate", "negative", "overflow"],
)
def test_bad_label_ids(tmp_path, label_ids, match):
    issues = _issues(tmp_path, n_datasets=1, label_ids=label_ids)
    assert len(issues) == 1
    assert "label_id" in issues[0]
    assert match in issues[0]


def test_missing_images_group_is_an_issue(tmp_path):
    path = write_collection(tmp_path / "c.zarr", n_datasets=1)
    del zarr.open_group(path)["dataset_0"]["images"]
    issues = validate(path)
    assert any("images must be exactly ['he']" in i for i in issues)
    del zarr.open_group(path)["dataset_0"]["labels"]
    assert any("no labels/cells" in i for i in validate(path))


def test_plain_group_is_an_issue(tmp_path):
    root = zarr.open_group(tmp_path / "c.zarr", mode="w", zarr_format=3)
    root.create_group("dataset_0")
    assert validate(tmp_path / "c.zarr") == ["dataset_0: the table has no uns['sddb_contract']"]
    assert validate(tmp_path / "missing.zarr")[0].startswith("collection: cannot open")


def test_table_without_obs_is_an_issue(tmp_path):
    path = write_collection(tmp_path / "c.zarr", n_datasets=1)
    del zarr.open_group(path)["dataset_0"]["obs"]
    issues = validate(path)
    assert len(issues) == 1
    assert issues[0].startswith("dataset_0: not an annbatch table or contract unreadable: KeyError")


def test_malformed_adapter_json_is_an_issue(tmp_path):
    path = write_collection(tmp_path / "c.zarr", n_datasets=1, contract={"adapters": ["{not json"]})
    issues = validate(path)
    assert len(issues) == 1
    assert "not an annbatch table or contract unreadable: JSONDecodeError" in issues[0]


def test_negative_counts(tmp_path):
    issues = _issues(tmp_path, overrides={0: {"negative_counts": True}}, n_cells=60)
    assert len(issues) == 1
    assert "negative" in issues[0]


def test_target_mpp_differs_across_datasets(tmp_path):
    finer = {"contract": {"mpp_target": 0.25, "mpp_native": 0.2}}
    issues = _issues(tmp_path, overrides={1: finer})
    assert len(issues) == 1
    assert issues[0].startswith("collection:")
    assert "mpp_target" in issues[0]


def test_stain_reference_roundtrips_through_zarr(tmp_path):
    from anndata.io import read_elem

    reference = {
        "stain_matrix": [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
        "white_point": [240.0, 238.0, 241.0],
        "max_concentrations": [1.9, 1.0, 0.5],
    }
    path = write_collection(tmp_path / "c.zarr", n_datasets=1, stain_reference=reference)
    assert validate(path) == []
    uns = read_elem(zarr.open_group(path, mode="r")["dataset_0/uns/sddb_contract"])
    assert DatasetContract.from_uns(uns).stain_reference == reference


def test_concave_centroid_lies_outside_its_mask():
    # The premise of the next two tests: a crescent's centre of mass is not inside the crescent.
    labels = draw_cells(200, 200, 280, crescents=1)
    cy, cx = ndimage.center_of_mass(labels == 1)
    assert labels[round(cy), round(cx)] != 1


def test_concave_cells_pass_r2(tmp_path):
    # 1 crescent in 200 cells (0.5%) is within the 99% tolerance.
    assert _issues(tmp_path, height=200, width=280, n_cells=200, crescents=1) == []


def test_centroids_outside_labels_fail_over_1pct(tmp_path):
    # 3 crescents in 200 cells (1.5%) are not.
    issues = _issues(tmp_path, height=200, width=280, n_cells=200, crescents=3)
    assert len(issues) == 2  # both datasets
    assert all("99%" in i and "concave" in i for i in issues)


def test_non_integer_counts(tmp_path):
    issues = _issues(tmp_path, overrides={0: {"fractional_counts": True}})
    assert len(issues) == 1
    assert "X" in issues[0]
    assert "integ" in issues[0]


def test_var_mismatch_across_datasets(tmp_path):
    names = [f"ENSG{i:011d}" for i in range(2, 22)]
    issues = _issues(tmp_path, overrides={1: {"var_names": names}})
    assert len(issues) == 1
    assert "var" in issues[0]
    assert "dataset_1" in issues[0]


def test_var_not_ensembl(tmp_path):
    issues = _issues(tmp_path, var_names=[f"GENE{i}" for i in range(20)])
    assert len(issues) == 2
    assert all("^ENS" in i for i in issues)


def test_mixed_grids_refused(tmp_path):
    native = {"contract": {"grid": "native", "mpp_target": None}}
    issues = _issues(tmp_path, overrides={1: native})
    assert len(issues) == 1
    assert "allow_mixed_grids" in issues[0]
    allowed = {"contract": {"allow_mixed_grids": True}}
    assert (
        _issues(tmp_path, overrides={0: allowed, 1: {"contract": {**native["contract"], "allow_mixed_grids": True}}})
        == []
    )


def test_upsampled_without_flag(tmp_path):
    issues = _issues(tmp_path, overrides={0: {"contract": {"upsampled": True}}})
    assert len(issues) == 1
    assert "allow_upsample" in issues[0]
    # A target finer than the native grid is upsampling, by any factor, whatever `upsampled` says.
    issues = _issues(tmp_path, overrides={0: {"contract": {"mpp_native": 0.505}}})
    assert len(issues) == 1
    assert "upsampl" in issues[0]


def test_mixed_roles_refused(tmp_path):
    issues = _issues(tmp_path, overrides={1: {"role": "if"}})
    assert issues == ["collection: one image role per collection, got ['he', 'if']"]


def test_transform_attr_refused(tmp_path):
    path = write_collection(tmp_path / "c.zarr", n_datasets=1)
    zarr.open_group(path)["dataset_0/labels/cells"].attrs["coordinateTransformations"] = [{"type": "scale"}]
    issues = validate(path)
    assert len(issues) == 1
    assert "labels/cells" in issues[0]
    assert "identity" in issues[0]


def test_extra_contract_key_forbidden(tmp_path):
    issues = _issues(tmp_path, overrides={0: {"contract": {"tile_px": 224}}})
    assert len(issues) == 1
    assert "tile_px" in issues[0]


def test_missing_label_id_column(tmp_path):
    path = write_collection(tmp_path / "c.zarr", n_datasets=1)
    obs = zarr.open_group(path)["dataset_0"]["obs"]
    del obs["label_id"]
    obs.attrs["column-order"] = [c for c in obs.attrs["column-order"] if c != "label_id"]
    assert any("label_id" in i for i in validate(path))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"grid": "target", "mpp_target": None}, "mpp_target"),
        ({"grid": "target", "mpp_native": None}, "mpp_native"),
        ({"upsampled": True}, "allow_upsample"),
        ({"role": "if", "channels": None}, "channels"),
    ],
)
def test_model_validators(change, match):
    with pytest.raises(ValueError, match=match):
        DatasetContract.model_validate({**base_contract(), **change})


def test_uns_roundtrip():
    contract = DatasetContract.model_validate(
        {**base_contract(), "stain_reference": {"stain_matrix": [[0.6, 0.7, 0.3], [0.1, 0.9, 0.4]], "method": "x"}}
    )
    assert DatasetContract.from_uns(contract.to_uns()) == contract


def test_import_sddb_does_not_import_pydantic_stack():
    code = (
        "import sys, sddb; "
        "print(sorted(m for m in ('pydantic', 'squidpy', 'spatialbatch', 'annbatch', 'dask_image', 'sddb.contract', 'sddb.loader') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_spatialbatch_loader_uses_validator(tmp_path):
    spatialbatch = pytest.importorskip("spatialbatch")
    Loader, SpatialDatasetCollection = spatialbatch.Loader, spatialbatch.SpatialDatasetCollection

    good = SpatialDatasetCollection(write_collection(tmp_path / "good.zarr", n_datasets=1))
    assert Loader(images=["he"], crop_px=16).use_collection(good).n_obs == 60
    bad = SpatialDatasetCollection(write_collection(tmp_path / "bad.zarr", n_datasets=1, image_dtype=np.uint16))
    with pytest.raises(ValueError, match="uint8"):
        Loader(images=["he"], crop_px=16).use_collection(bad)


def test_absent_cells_allowed_up_to_the_centroid_fallback_count(tmp_path):
    knobs = {"height": 200, "width": 280, "n_cells": 200, "n_datasets": 1}
    assert (
        _issues(tmp_path / "a", **knobs, overrides={0: {"erase_cells": 1, "contract": {"n_centroid_fallback": 1}}})
        == []
    )
    issues = _issues(tmp_path / "b", **knobs, overrides={0: {"erase_cells": 2, "contract": {"n_centroid_fallback": 1}}})
    assert len(issues) == 1
    assert "allows 1 (n_centroid_fallback)" in issues[0]


@pytest.mark.parametrize("value", [-1, 1.5, "2"])
def test_bad_centroid_fallback_count(tmp_path, value):
    issues = _issues(tmp_path, n_datasets=1, overrides={0: {"contract": {"n_centroid_fallback": value}}})
    assert len(issues) == 1
    assert "n_centroid_fallback" in issues[0]


def test_contract_without_centroid_fallback_field_still_validates(tmp_path):
    """n_centroid_fallback is optional: a uns without the key reads as 0 and validates."""
    path = write_collection(tmp_path / "c.zarr", n_datasets=1)
    uns = zarr.open_group(path / "dataset_0" / "uns" / "sddb_contract", mode="a")
    del uns["n_centroid_fallback"]
    assert "n_centroid_fallback" not in uns
    assert validate(path) == []
    assert DatasetContract.from_uns(base_contract()).n_centroid_fallback == 0


def _features(n_controls=2, present=None):
    """Var columns: GE Ensembl ids first, then `n_controls` controls; `present` as given."""
    types = ["Gene Expression"] * (N_VARS - n_controls) + ["Negative Control Probe"] * n_controls
    names = VAR_NAMES[: N_VARS - n_controls] + [f"NegCtrl_{i}" for i in range(n_controls)]
    columns = {"feature_types": types}
    if present is not None:
        columns["present"] = present
    return {"var_names": names, "var_columns": columns}


def test_ensembl_ids_required_only_for_gene_expression(tmp_path):
    assert _issues(tmp_path / "a", n_datasets=1, **_features()) == []
    knobs = _features()
    knobs["var_columns"]["feature_types"] = ["Gene Expression"] * N_VARS  # the controls marked as genes
    issues = _issues(tmp_path / "b", n_datasets=1, **knobs)
    assert len(issues) == 1
    assert "2 Gene Expression names not matching ^ENS" in issues[0]


def test_presence_mask_agrees_with_contract_and_x(tmp_path):
    present = [True] * (N_VARS - 1) + [False]
    counts = {"n_features_present": N_VARS - 1, "n_features_filled": 1}
    ok = {**_features(present=present), "zero_features": [N_VARS - 1], "contract": counts}
    assert _issues(tmp_path / "a", n_datasets=1, **ok) == []
    issues = _issues(tmp_path / "b", n_datasets=1, **{**ok, "zero_features": []})
    assert len(issues) == 1
    assert "values != 0 in the sampled rows for features var['present'] says are filled" in issues[0]
    issues = _issues(tmp_path / "c", n_datasets=1, **{**ok, "contract": {**counts, "n_features_filled": 0}})
    assert issues == ["dataset_0: the contract's n_features_filled is 0, var['present'] gives 1"]
    issues = _issues(tmp_path / "d", n_datasets=1, **{**_features(), "contract": counts})
    assert len(issues) == 1
    assert "var has no 'present' column" in issues[0]


@pytest.mark.parametrize("field", ["n_features_present", "n_features_filled"])
@pytest.mark.parametrize("value", [-1, 1.5, "2"])
def test_bad_feature_count(tmp_path, field, value):
    issues = _issues(tmp_path, n_datasets=1, overrides={0: {"contract": {field: value}}})
    assert len(issues) == 1
    assert field in issues[0]


def test_contract_without_feature_count_fields_still_validates(tmp_path):
    path = write_collection(tmp_path / "c.zarr", n_datasets=1)
    uns = zarr.open_group(path / "dataset_0" / "uns" / "sddb_contract", mode="a")
    for field in ("n_features_present", "n_features_filled"):
        del uns[field]
    assert validate(path) == []
    contract = DatasetContract.from_uns(base_contract())
    assert (contract.n_features_present, contract.n_features_filled) == (0, 0)


def test_encoding_before_crop_at_read_is_refused(tmp_path):
    path = write_collection(tmp_path / "c.zarr", n_datasets=1)
    group = zarr.open_group(path / "dataset_0", mode="a")
    group.attrs["spatialbatch"] = {**group.attrs["spatialbatch"], "encoding-version": "0.1.0"}
    issues = validate(path)
    assert len(issues) == 1
    assert "encoding-version" in issues[0]
    assert "0.2.0" in issues[0]


def test_ensembl_ids_checked_only_where_a_gene_has_one(tmp_path):
    ids = [*VAR_NAMES[:-2], None, VAR_NAMES[0]]  # one Gene Expression row without an id, one repeated id
    names = [*VAR_NAMES[:-2], "NOID", f"{VAR_NAMES[0]}-1"]
    knobs = {"var_names": names, "var_columns": {"feature_types": ["Gene Expression"] * N_VARS, "gene_ids": ids}}
    assert _issues(tmp_path / "a", n_datasets=1, **knobs) == []
    bad = {**knobs, "var_columns": {**knobs["var_columns"], "gene_ids": [*ids[:-1], "X123"]}}
    issues = _issues(tmp_path / "b", n_datasets=1, **bad)
    assert len(issues) == 1
    assert "var['gene_ids'] has 1 Gene Expression names not matching ^ENS" in issues[0]


def test_gene_key_counts_must_match_the_var(tmp_path):
    ids = [*VAR_NAMES[:-2], None, VAR_NAMES[0]]
    names = [*VAR_NAMES[:-2], "NOID", f"{VAR_NAMES[0]}-1"]
    columns = {"feature_types": ["Gene Expression"] * N_VARS, "gene_ids": ids, "present": [True] * N_VARS}

    def record(**params):
        rec = {"name": "GeneKey", "version": "0.3.0", "params": {"source": N_VARS, **params}}
        return {"adapters": [json.dumps(rec, sort_keys=True)], "n_features_present": N_VARS}

    knobs = {"var_names": names, "var_columns": columns}
    assert _issues(tmp_path / "a", n_datasets=1, **knobs, contract=record(no_ensembl_id=1, duplicates=1)) == []
    issues = _issues(tmp_path / "b", n_datasets=1, **knobs, contract=record(no_ensembl_id=0, duplicates=2))
    assert sorted(i.split(": ", 1)[1] for i in issues) == [
        "the GeneKey record's duplicates is 2, var gives 1",
        "the GeneKey record's no_ensembl_id is 0, var gives 1",
    ]
    rec = record(no_ensembl_id=1, duplicates=1)
    rec["adapters"] = [rec["adapters"][0].replace(f'"source": {N_VARS}', f'"source": {N_VARS + 3}')]
    issues = _issues(tmp_path / "c", n_datasets=1, **knobs, contract=rec)
    assert issues == [
        f"dataset_0: the GeneKey record's source is {N_VARS + 3}, var gives {N_VARS} (a source feature was dropped)"
    ]


def test_flagged_centroid_outside_the_grid_validates(tmp_path):
    """A cell beyond the image keeps its true centroid when obs['outside_image'] flags it; R2 skips it."""
    knobs = {"n_datasets": 1}
    flagged = {"move_centroid": (-5.0, 10.0), "outside_image": [0], "contract": {"n_outside_image": 1}}
    assert _issues(tmp_path / "a", **knobs, overrides={0: flagged}) == []
    # unflagged: the out-of-grid centre fails
    issues = _issues(tmp_path / "b", **knobs, overrides={0: {"move_centroid": (-5.0, 10.0)}})
    assert any("outside the image" in i and "does not flag" in i for i in issues)
    # the count must match the flags
    bad = {**flagged, "contract": {"n_outside_image": 2}}
    assert any("n_outside_image is 2" in i for i in _issues(tmp_path / "c", **knobs, overrides={0: bad}))
    # a count without the column fails
    issues = _issues(tmp_path / "d", **knobs, overrides={0: {"contract": {"n_outside_image": 1}}})
    assert any("no column 'outside_image'" in i for i in issues)


@pytest.mark.parametrize("value", [-1, 1.5, "2"])
def test_bad_outside_image_count(tmp_path, value):
    issues = _issues(tmp_path, n_datasets=1, overrides={0: {"contract": {"n_outside_image": value}}})
    assert any("n_outside_image" in i for i in issues)
