from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests._fixtures import write_fixture_catalog, write_fixture_feature_index, write_fixture_gene_index

from sddb import Catalog


def test_lookups_never_load_the_index(tmp_path, monkeypatch):
    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    write_fixture_gene_index(tmp_path / "gene_index.parquet")
    write_fixture_feature_index(tmp_path / "feature_index.parquet")
    real = pd.read_parquet

    def guarded(path, *a, **kw):
        if "_index" in str(path):
            raise AssertionError(f"index loaded whole: {path}")
        return real(path, *a, **kw)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    cat = Catalog(cat_p.as_uri(), cache_dir=tmp_path / "cache")
    assert list(cat.genes.ranked("EPCAM")["uid"]) == ["uid0001", "uid0002", "uid0005"]
    assert sorted(d.uid for d in cat.genes.where_expressed(["epcam", "ENSG00000170421"])) == ["uid0001"]
    assert cat.genes.resolve([]) == {}
    assert sorted(d.uid for d in cat.features.where_measured(["EPCAM", "KRT8"], mode="any")) == [
        "uid0001",
        "uid0002",
        "uid0005",
    ]
    hits = cat.features.where_measured(["EPCAM", "p16422"], mode="any")  # label + list-column ref in one batch
    assert sorted(d.uid for d in hits) == ["uid0001", "uid0002", "uid0003", "uid0005"]
    assert sorted(d.uid for d in cat.features.by_gene("EPCAM")) == ["uid0001", "uid0002", "uid0003", "uid0005"]


_MEASURE = """
import resource, sys, time
from sddb import Catalog
cat = Catalog(sys.argv[1], cache_dir=sys.argv[2])
cat.to_df()
before, t = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, time.perf_counter()
n = len(cat.genes.ranked("EPCAM"))
dt = time.perf_counter() - t
scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
print(n, (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * scale, dt)
"""


@pytest.mark.slow
def test_gene_lookup_memory_at_scale(tmp_path):
    """~10M-row gene index (60k symbols, 6k uids): one lookup grows peak RSS by well under 1.5 GB."""
    rng = np.random.default_rng(0)
    n, n_sym, n_uid = 10_000_000, 60_000, 6_000
    syms = pa.array(["EPCAM", *(f"G{i}" for i in range(1, n_sym))])
    fids = pa.array([f"ENSG{i:011d}" for i in range(n_sym)])
    uids = pa.array([f"uid{i:04d}" for i in range(1, n_uid + 1)])  # uid0001..uid0005 are catalog datasets
    writer = None
    for i in range(n // 1_000_000):
        s = rng.integers(0, n_sym, 1_000_000, dtype=np.int32)
        u = rng.integers(0, n_uid, 1_000_000, dtype=np.int32)
        if i == 5:
            s[:5], u[:5] = 0, np.arange(5)  # EPCAM in each catalog dataset
        table = pa.table(
            {
                "symbol": pa.DictionaryArray.from_arrays(s, syms),
                "feature_id": pa.DictionaryArray.from_arrays(s, fids),
                "uid": pa.DictionaryArray.from_arrays(u, uids),
                "total_counts": rng.random(1_000_000) * 1000,
                "fraction_obs_detected": rng.random(1_000_000),
            }
        )
        writer = writer or pq.ParquetWriter(tmp_path / "gene_index.parquet", table.schema)
        writer.write_table(table)
    writer.close()
    cat_p = write_fixture_catalog(tmp_path / "catalog.parquet")
    out = subprocess.run(
        [sys.executable, "-c", _MEASURE, cat_p.as_uri(), str(tmp_path / "cache")], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    hits, growth, seconds = out.stdout.split()
    print(f"rows={n} hits={hits} peak_rss_growth={int(growth) / 2**20:.0f} MiB time={float(seconds):.1f} s")
    assert int(hits) == 5
    assert int(growth) < 1.5 * 2**30


def test_fold_match_is_exact_casefold():
    from sddb.features import _fold_match

    values = pa.array(["EPCAM", "Straße", None, "ÉCLAIR", "krt19"])
    assert _fold_match(values, {"epcam", "strasse", "éclair"}) == {0: "epcam", 1: "strasse", 3: "éclair"}
