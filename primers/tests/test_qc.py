import anndata as ad
import numpy as np
import pandas as pd
import pytest
from primers.qc import apply_qc, floor_loss, load_thresholds

T = {
    "spot": {
        "technologies": ["Visium"],
        "min_counts": 100,
        "min_genes": 10,
        "max_pct_mt": 20,
        "flag": {"min_median_counts": 500, "min_share_kept": 0.5},
        "reason": "r",
    },
    "imaging_cell": {
        "technologies": ["Xenium"],
        "min_counts": 10,
        "min_genes": 3,
        "flag": {"min_median_counts": 20, "min_share_kept": 0.5},
        "reason": "r",
    },
}
GENES = [f"G{i}" for i in range(58)] + ["MT-CO1", "MT-ND1"]


def _cohort(parts):
    """parts: (uid, technology, per-cell total count, n cells, mt share); counts spread over the first 30 genes."""
    blocks = []
    for uid, tech, total, n, mt in parts:
        X = np.zeros((n, len(GENES)), "float32")
        X[:, :30] = total * (1 - mt) / 30
        X[:, -2:] = total * mt / 2
        obs = pd.DataFrame({"dataset_uid": uid, "technology": tech}, index=[f"{uid}-{i}" for i in range(n)])
        blocks.append(ad.AnnData(X, obs=obs, var=pd.DataFrame(index=GENES)))
    return ad.concat(blocks)


def test_thresholds_applied_per_technology():
    """A 50-count cell passes the imaging floor (10) and fails the spot floor (100)."""
    out, qc = apply_qc(_cohort([("v", "Visium", 50, 5, 0), ("x", "Xenium", 50, 5, 0)]), T)
    q = qc.set_index("dataset_uid")
    assert q.loc["v", ["n_before", "n_after"]].tolist() == [5, 0]
    assert q.loc["x", ["n_before", "n_after"]].tolist() == [5, 5]
    assert set(out.obs["dataset_uid"]) == {"x"}


def test_flag_when_outside_bounds():
    """Median counts under the flag bound flag a dataset; a high mitochondrial share removes its observations.

    The "mt" dataset loses every observation to the mitochondrial cap, so the kept-share bound flags it.
    """
    _, qc = apply_qc(
        _cohort([("ok", "Visium", 1000, 4, 0.05), ("low", "Visium", 200, 4, 0.05), ("mt", "Visium", 1000, 4, 0.5)]), T
    )
    q = qc.set_index("dataset_uid")
    assert not q.loc["ok", "flagged"]
    assert q.loc["low", "flagged"]
    assert "median counts" in q.loc["low", "flag_reason"]
    assert q.loc["mt", "n_after"] == 0
    assert q.loc["mt", "flagged"]
    assert "kept <" in q.loc["mt", "flag_reason"]
    assert q.loc["ok", "pct_mt_median"] == pytest.approx(5.0)


def test_qc_median_is_before_filtering():
    """The median counts are taken over all observations, before the filters remove any."""
    low, high = _cohort([("v", "Visium", 50, 3, 0)]), _cohort([("v", "Visium", 1000, 2, 0)])
    high.obs_names = [f"v-high-{i}" for i in range(2)]
    out, qc = apply_qc(ad.concat([low, high]), T)
    assert out.n_obs == 2  # filtering keeps only the 1000-count spots, whose median is 1000
    assert qc.set_index("dataset_uid").loc["v", "median_counts"] == pytest.approx(50)


def test_control_probes_left_out_of_counts():
    """NegControlProbe_* and BLANK_* features do not count towards total_counts or n_genes."""
    a = _cohort([("x", "Xenium", 60, 4, 0)])
    probes = ad.AnnData(
        np.full((4, 2), 7, "float32"), obs=a.obs, var=pd.DataFrame(index=["NegControlProbe_00001", "BLANK_0001"])
    )
    out, _ = apply_qc(ad.concat([a, probes], axis=1, merge="first"), T)
    assert out.obs["total_counts"].tolist() == [60] * 4
    assert out.obs["n_genes"].tolist() == [30] * 4
    assert out.obs["total_counts"].dtype.kind == "i"
    assert out.obs["n_genes"].dtype.kind == "i"


def test_zero_cells_after_qc_marks_dropped():
    """A dataset with no cell left is marked dropped, not silently missing."""
    _, qc = apply_qc(_cohort([("v", "Visium", 50, 5, 0), ("x", "Xenium", 50, 5, 0)]), T)
    q = qc.set_index("dataset_uid")
    assert q.loc["v", "status"] == "dropped: no cells after QC"
    assert q.loc["x", "status"] == "kept"


def test_refuses_fewer_than_50_shared_genes():
    """QC on fewer than 50 genes is refused, because the per-cell metrics would mean little."""
    adata = _cohort([("x", "Xenium", 50, 5, 0)])[:, :40]
    with pytest.raises(ValueError, match="40 genes"):
        apply_qc(adata, T)


def test_floor_loss_reported_per_technology():
    """The share lost at a count floor, per technology, optionally on a subset of genes (gate 3)."""
    adata = _cohort([("v", "Visium", 1000, 4, 0), ("x", "Xenium", 15, 4, 0)])
    assert floor_loss(adata, 100) == {"Visium": 0.0, "Xenium": 1.0}
    # on 3 of the 30 expressed genes, a 1000-count Visium spot keeps 100 counts
    assert floor_loss(adata, 100, genes=["G0", "G1", "G2"]) == {"Visium": 0.0, "Xenium": 1.0}
    assert floor_loss(adata, 101, genes=["G0", "G1", "G2"])["Visium"] == 1.0


def test_shipped_thresholds_cover_every_family_technology():
    """thresholds.yaml covers every technology of every family, with a reason per group."""
    from primers.cohort import FAMILIES

    t = load_thresholds()
    techs = {x for g in t.values() for x in g["technologies"]}
    assert {x for v in FAMILIES.values() for x in v} <= techs
    assert all(g["reason"] for g in t.values())


def test_floor_loss_per_technology_floor():
    """A per-technology floor mapping applies each technology's own floor."""
    from primers.qc import qc_floors

    adata = _cohort([("v", "Visium", 1000, 4, 0), ("x", "Xenium", 15, 4, 0)])
    assert floor_loss(adata, {"Visium": 2000, "Xenium": 10}) == {"Visium": 1.0, "Xenium": 0.0}
    assert qc_floors(T) == {"Visium": 100.0, "Xenium": 10.0}
