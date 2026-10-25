"""Tests for role -> element path resolution."""

from __future__ import annotations

import pytest

from sddb.roles import resolve

ROLES = {
    "he_full": r"^images/.*_full_image$",
    "bins": r"^tables/square_008um$",
    "bin_shapes": r"^shapes/.*_square_008um$",
}


def _names(prefix: str) -> list[str]:
    return [
        f"images/{prefix}_full_image",
        f"images/{prefix}_hires_image",
        "tables/square_008um",
        "tables/square_016um",
        f"shapes/{prefix}_square_008um",
        f"shapes/{prefix}_square_016um",
    ]


def test_unique_match():
    got = resolve(_names("Visium_HD_Human_Colon_Cancer_P1"), ROLES)
    assert got == {
        "he_full": "images/Visium_HD_Human_Colon_Cancer_P1_full_image",
        "bins": "tables/square_008um",
        "bin_shapes": "shapes/Visium_HD_Human_Colon_Cancer_P1_square_008um",
    }


def test_other_prefix():
    got = resolve(_names("crc_p1"), ROLES)
    assert got["he_full"] == "images/crc_p1_full_image"
    assert got["bin_shapes"] == "shapes/crc_p1_square_008um"


def test_no_match_raises_keyerror_naming_role():
    with pytest.raises(KeyError) as e:
        resolve(_names("x"), {"he_lowres": r"^images/.*_lowres_image$"})
    msg = str(e.value)
    assert "he_lowres" in msg
    assert "images/x_full_image" in msg
    assert "images/x_hires_image" in msg
    assert "tables/square_008um" not in msg


def test_ambiguous_raises_valueerror():
    with pytest.raises(ValueError, match="bins") as e:
        resolve(_names("x"), {"bins": r"^tables/square_"})
    assert "tables/square_008um" in str(e.value)
    assert "tables/square_016um" in str(e.value)


def test_dataset_resolve(monkeypatch):
    import pandas as pd

    from sddb import Dataset, remote

    monkeypatch.setattr(remote, "elements", lambda url: dict.fromkeys(_names("crc_p1"), {}))
    ds = Dataset(pd.Series({"uid": "u1", "zarr_url": "s3://b/1.zarr"}))
    assert ds.resolve(ROLES)["bins"] == "tables/square_008um"
