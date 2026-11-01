"""Cohort helpers: the batch variable ``paper`` for cohort members, and loading a cohort.

The rule resolves ``paper`` at the series or collection level, not per row:

1. Group members by GEO series (a ``geo-gse*`` entry in ``collections``), else by their first
   collection slug, else by ``study_id``.
2. If any member of a group has a DOI, every member gets it. Members without their own DOI get
   ``paper_source == "doi_inferred"`` and ``paper_inferred_from`` names the uid that carried it.
3. Otherwise ``paper`` is the group key (GEO series, collection slug or study_id).
4. Every 10x Genomics demo release (a key that starts with ``10x``) counts as one paper,
   ``10x Genomics demo datasets``, with ``paper_source == "10x_demo"``. They come from one company,
   not from independent labs.

A DOI inferred from a member that is later dropped (for its licence, say) still counts: it names the
same group, so ``paper_inferred_from`` can name a dropped uid. A 10x release with its own DOI keeps that
DOI as its paper and is not merged with the demos.

Keys are lowercased with ``:`` read as ``-``, so a study id such as ``10x:foo`` and the collection
slug ``10x-foo`` count as one paper. A member with no DOI, collections or study_id has no paper:
``paper`` is missing and ``paper_source`` is ``none``. It never becomes a paper of its own.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import anndata as ad
import numpy as np
import pandas as pd


def _collections(value: object) -> list[str]:
    if value is None or isinstance(value, str) or not isinstance(value, Iterable):
        return [value] if isinstance(value, str) and value else []
    return [str(v) for v in value if v is not None and str(v)]


def _slug(key: str) -> str:
    # study ids ("10x:foo", "geo:GSE1") and collection slugs ("10x-foo", "geo-gse1") name the same source.
    return key.strip().lower().replace(":", "-")


TENX_DEMO_PAPER = "10x Genomics demo datasets"
NO_PAPER = "dropped: no paper (no study_id, series or collection)"
TENX_RERELEASE = "dropped: 10x re-release of a section kept in a newer release"
_TENX_RELEASE = r"-(\d+)-(\d+)-(\d+)$"  # trailing Space Ranger release, e.g. -1-0-0
_TENX_SECTION = r"-section-\d+-.*$"  # serial sections of one block: "<block>-section-<n>-..."


def tenx_sections(keys: pd.Series) -> tuple[pd.Series, pd.Series]:
    """``(is_rerelease, block)`` for the members' source keys (paper dedup rule A).

    10x publishes one section in several Space Ranger releases (``...-1-0-0``, ``...-1-1-0``). Per section
    the newest release is kept, as in the catalog's own QC, and the others are re-releases; a key without a
    release suffix counts as newer than every numbered one. ``block`` is the section's key without the release and without
    ``-section-<n>-...``, so serial sections of one block share it. Keys not from 10x get ``False`` and a
    missing block.
    """
    keys = keys.fillna("").astype(str)
    is10 = keys.str.startswith("10x")
    stem = keys.str.replace(_TENX_RELEASE, "", regex=True)
    release = keys.str.extract(_TENX_RELEASE).astype(float).fillna(float("inf"))
    order = pd.DataFrame({"stem": stem, "r0": release[0], "r1": release[1], "r2": release[2]})[is10]
    first = order.sort_values(["r0", "r1", "r2"], ascending=False).drop_duplicates("stem").index
    rerelease = is10 & ~keys.index.isin(first)
    block = stem.str.replace(_TENX_SECTION, "", regex=True).where(is10)
    return pd.Series(rerelease, index=keys.index), block


def _group(row: pd.Series) -> tuple[str | None, str]:
    """Return ``(group key, source)`` for one member, or ``(None, "none")`` when it has no key."""
    cols = [_slug(c) for c in _collections(row.get("collections"))]
    geo = sorted(c for c in cols if c.startswith("geo-gse"))
    if geo:
        return geo[0], "geo_series"
    if cols:
        return cols[0], "collection"
    study = row.get("study_id")
    if pd.isna(study) or not str(study).strip():
        return None, "none"
    return _slug(str(study)), "study_id"


def _doi(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def paper_for_members(members: pd.DataFrame, *, merge_10x_demos: bool = True) -> pd.DataFrame:
    """Add ``paper``, ``paper_source`` and ``paper_inferred_from`` columns by the cohort rule.

    ``paper_source`` is one of ``doi``, ``doi_inferred``, ``geo_series``, ``collection``, ``study_id``,
    ``10x_demo`` or ``none`` (no paper; ``paper`` is missing). ``paper_inferred_from`` is the uid whose
    DOI was propagated, or ``""``. ``merge_10x_demos=False`` keeps each 10x demo release as its own
    paper, which the cohort report uses to show both counts.
    """
    out = members.copy()
    if out.empty:
        return out.assign(
            paper=pd.Series(dtype=object),
            paper_source=pd.Series(dtype=object),
            paper_inferred_from=pd.Series(dtype=object),
        )
    grouped = [_group(row) for _, row in out.iterrows()]
    keys = pd.Series([g[0] for g in grouped], index=out.index)
    group_source = pd.Series([g[1] for g in grouped], index=out.index)
    dois = out["doi"].map(_doi) if "doi" in out else pd.Series([None] * len(out), index=out.index)

    # Per group: the most common DOI and the first uid that carries it.
    carrier: dict[str, tuple[str, str]] = {}
    for key, idx in keys.groupby(keys).groups.items():
        present = dois.loc[idx].dropna()
        if present.empty:
            continue
        doi = present.value_counts().index[0]
        uid = str(out.loc[present[present == doi].index[0], "uid"])
        carrier[key] = (doi, uid)

    paper, source, inferred = [], [], []
    for i in out.index:
        own, key = dois.loc[i], keys.loc[i]
        if not pd.isna(own):
            paper.append(own)
            source.append("doi")
            inferred.append("")
        elif key in carrier:
            paper.append(carrier[key][0])
            source.append("doi_inferred")
            inferred.append(carrier[key][1])
        elif merge_10x_demos and isinstance(key, str) and key.startswith("10x"):
            paper.append(TENX_DEMO_PAPER)
            source.append("10x_demo")
            inferred.append("")
        else:
            paper.append(key)
            source.append(group_source.loc[i])
            inferred.append("")
    out["paper"] = paper
    out["paper_source"] = source
    out["paper_inferred_from"] = inferred
    return out


# Technology families. Each family is joined on its own, because an outer join across families would
# pad the imaging panels with zeros for every spot-only gene.
FAMILIES: dict[str, list[str]] = {
    "spot": ["Visium", "VisiumHD", "SpatialTranscriptomics"],
    "imaging_cell": ["Xenium", "Merscope", "MERFISH"],
}
MIN_PAPERS = 3
GENE_KEY = "symbol"  # every source carries symbols; Allen MERSCOPE and MERFISH tables may lack Ensembl ids
_SPEC_FAMILY_ALIASES = {"imaging": "imaging_cell"}
NO_SYMBOLS = "dropped: load error: no gene symbols"
OBS_COLUMNS = (
    "dataset_uid",
    "block",
    "paper",
    "paper_source",
    "technology",
    "tissue",
    "donor_id",
    "license",
    "license_noncommercial",
)
TABLE_COLUMNS = (
    "uid",
    "title",
    "technology",
    "family",
    "paper",
    "paper_source",
    "paper_inferred_from",
    "license",
    "license_noncommercial",
    "n_obs",
    "n_genes",
    "n_ensembl",
    "status",
)


def _true(value: object) -> bool:
    return value is not None and not pd.isna(value) and bool(value)


def spec_families(spec: dict, families: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """The families a cohort spec uses: all of ``families`` (default ``FAMILIES``), or those it lists.

    ``imaging`` in a spec means ``imaging_cell``.
    """
    families = dict(families or FAMILIES)
    if wanted := spec.get("families"):
        names = {_SPEC_FAMILY_ALIASES.get(f, f) for f in wanted}
        families = {k: v for k, v in families.items() if k in names}
    return families


def family_query(catalog: object, spec: dict, families: dict[str, list[str]]) -> dict:
    """``spec["query"]``, limited to the family technologies this catalog has when it names no technology.

    ``Catalog.query`` refuses a technology value that is not in the catalog, so absent ones are left out.
    """
    query = dict(spec.get("query") or {})
    if "technology" not in query:
        present = set(catalog.facets("technology"))
        query["technology"] = [t for techs in families.values() for t in techs if t in present]
    return query


def _candidates(catalog: object, spec: dict, families: dict[str, list[str]]) -> pd.DataFrame:
    """The query rows plus ``include_uids``, minus ``exclude_uids``. No licence filter yet."""
    df = catalog.query(**family_query(catalog, spec, families)).to_df()
    if include := list(spec.get("include_uids") or []):
        df = pd.concat([df, catalog.cohort(uids=include).to_df()]).drop_duplicates("uid")
    exclude = {str(u) for u in spec.get("exclude_uids") or []}
    return df[~df["uid"].astype(str).isin(exclude)].reset_index(drop=True)


def cohort_members(catalog: object, spec: dict, families: dict[str, list[str]]) -> pd.DataFrame:
    """The cohort's candidate members with ``paper``, ``family``, ``dataset_uid`` and ``status``.

    ``status`` is ``kept`` or ``dropped: <reason>`` for the licence, a technology outside every family,
    a missing paper key, or a 10x re-release (``tenx_sections``). ``block`` is the 10x block of serial
    sections, else the uid. Tables are not read.
    """
    family_of = {t: fam for fam, techs in families.items() for t in techs}
    members = paper_for_members(_candidates(catalog, spec, families))
    members["dataset_uid"] = members["uid"].astype(str)
    members["family"] = members["technology"].map(family_of)
    status = pd.Series("kept", index=members.index, dtype=object)
    redistributable = members["license_redistributable"].map(_true)
    if "license_unknown" in members:
        redistributable &= ~members["license_unknown"].map(_true)
    status[~redistributable] = "dropped: licence not redistributable"
    status[(status == "kept") & members["family"].isna()] = "dropped: technology not in a family"
    status[(status == "kept") & members["paper"].isna()] = NO_PAPER
    key = members["canonical_key"] if "canonical_key" in members else pd.Series(None, index=members.index)
    rerelease, block = tenx_sections(key.fillna(members["study_id"]))
    status[(status == "kept") & rerelease] = TENX_RERELEASE
    members["block"] = block.fillna(members["dataset_uid"])
    members["status"] = status
    return members


WRONG_ORGANISM = "dropped: genes from another organism"
_GENOMES = {"human": {"grch38", "grch37", "hg38", "hg19"}, "mouse": {"grcm38", "grcm39", "mm10", "mm39"}}
_ENSEMBL = {"human": r"ENSG\d", "mouse": r"ENSMUSG\d"}
_ENSEMBL_ID = r"ENS[A-Z]*G\d{6}"  # an Ensembl gene id; ENSA is a gene symbol
_ID_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "gene_id_raw", "feature_id", "original_var_name")
_PREFIX = re.compile(r"^([A-Za-z0-9.]+?)_+(?=[A-Za-z0-9])")
_MIN_GENOME_GENES = 1000  # a prefix this common names a reference genome, not a gene family


class WrongOrganismError(ValueError):
    """The table has no genes of the dataset's catalog organism."""


def restrict_to_organism(adata: ad.AnnData, organism: str) -> ad.AnnData:
    """Keep only the genes of ``organism`` and strip genome prefixes from their names.

    Tables aligned to a combined reference (a xenograft, or a host plus a pathogen) carry genes of
    several species. A name prefix shared by at least 1000 genes marks a genome (``GRCh38__OR4F5``,
    ``hg38_SAMD11``, ``mm39_Xkr4``) or a pathogen's gene ids (``ANKA_...``, ``PBANKA_0101100``).

    1. When a prefix names the organism's genome, those genes are kept with the prefix removed.
    2. Otherwise, when at least 1000 genes carry no such prefix, they are taken as the host's genes and
       the prefixed ones are dropped.
    3. Then the Ensembl ids decide (``ENSG`` for human, ``ENSMUSG`` for mouse): genes with another
       organism's ids are dropped, and so are features without an Ensembl id when there are at least
       1000 of them (a pathogen genome); a few (a transgene, say) are kept.

    Organisms other than human and mouse are returned unchanged. Raises ``WrongOrganismError`` when no
    gene of the organism is left, e.g. a dataset labelled mouse whose table holds only human genes.
    """
    if organism not in _GENOMES:
        return adata
    names = pd.Series(adata.var_names.astype(str))
    prefix = names.str.extract(_PREFIX)[0].str.lower()
    counts = prefix.value_counts()
    genomes = set(counts[counts >= _MIN_GENOME_GENES].index)
    if genomes:
        own_genome = prefix.isin(genomes & _GENOMES[organism])
        unprefixed = ~prefix.isin(genomes)
        if own_genome.any():
            adata = adata[:, own_genome.to_numpy()].copy()
            adata.var_names = names[own_genome].str.replace(_PREFIX, "", regex=True).to_numpy()
        elif unprefixed.sum() >= _MIN_GENOME_GENES:
            adata = adata[:, unprefixed.to_numpy()].copy()
        else:
            raise WrongOrganismError(f"genome prefixes {sorted(genomes)}, none for {organism}")
    ids = next(
        (
            adata.var[c].astype(str)
            for c in _ID_COLUMNS
            if c in adata.var and adata.var[c].astype(str).str.match(_ENSEMBL_ID).mean() >= 0.5
        ),
        None,
    )
    if ids is None:
        return adata
    own = ids.str.match(_ENSEMBL[organism]).to_numpy(dtype=bool, na_value=False)
    other = ids.str.match("|".join(v for k, v in _ENSEMBL.items() if k != organism)).to_numpy(
        dtype=bool, na_value=False
    )
    if not own.any():
        raise WrongOrganismError(f"{int(other.sum())} genes of another organism and none of {organism}")
    no_id = ~(own | other)
    drop = other | (no_id if no_id.sum() >= _MIN_GENOME_GENES else np.zeros_like(no_id))
    return adata[:, ~drop].copy() if drop.any() else adata


def n_ensembl(adata: ad.AnnData) -> int:
    """The number of genes with an Ensembl gene id in any of the id columns of ``var``."""
    has = np.zeros(adata.n_vars, dtype=bool)
    for c in _ID_COLUMNS:
        if c in adata.var:
            has |= adata.var[c].astype(str).str.match(_ENSEMBL_ID).to_numpy(dtype=bool, na_value=False)
    return int(has.sum())


HD_TABLE, HD_FINE_TABLE = "square_016um", "square_008um"
# scripts/hd_aggregate_check.py showed that 2 x 2 sums of 8 um bins reproduce Space Ranger's 16 um table (the
# Lead's condition, 2026-11-01): on iAZtgjxeI9UbeR4w0000 (HPC job 42384010) all 122,274 bins matched, 99.4%
# exactly, total counts within 0.04%, correlation of bin totals 1.0. Set False to drop such stores again.
HD_SUM_CHECKED = True
NO_HD_16 = "dropped: no 16 um table, and the 2 x 2 sum of 8 um bins is not yet checked"


class NoHD16Error(ValueError):
    """A Visium HD store without a 16 um table, while the 2 x 2 sum is not yet checked."""


def aggregate_bins(adata: ad.AnnData, factor: int = 2) -> ad.AnnData:
    """Sum ``factor`` x ``factor`` blocks of grid bins (``obs["array_row"]``, ``obs["array_col"]``) into one bin.

    Used to build 16 µm Visium HD bins from 8 µm bins when a store has no 16 µm table. Counts are summed,
    ``obsm["spatial"]`` is the mean of the block, and the other obs columns come from the block's first bin.
    """
    import scipy.sparse as sp

    row = adata.obs["array_row"].to_numpy(dtype=np.int64) // factor
    col = adata.obs["array_col"].to_numpy(dtype=np.int64) // factor
    codes, uniques = pd.factorize(row * (int(col.max()) + 1) + col)
    n_new = len(uniques)
    member = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(n_new, adata.n_obs))
    first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    obs = adata.obs.iloc[first].copy()
    obs["array_row"], obs["array_col"] = row[first], col[first]
    obs.index = [f"{i}-{factor}x" for i in obs.index]
    X = sp.csr_matrix(member @ sp.csr_matrix(adata.X))
    out = ad.AnnData(X=X.astype(np.float32), obs=obs, var=adata.var.copy())
    if "spatial" in adata.obsm:
        sizes = np.asarray(member.sum(axis=1)).ravel()[:, None]
        out.obsm["spatial"] = np.asarray(member @ np.asarray(adata.obsm["spatial"], dtype=float)) / sizes
    return out


def hd_table(stored: list[str] | set[str]) -> str:
    """The Visium HD table a store's elements offer: the 16 µm table, else the 8 µm one (summed 2 x 2 on load)."""
    return HD_TABLE if f"tables/{HD_TABLE}" in stored else HD_FINE_TABLE


def _load_member(row: pd.DataFrame, obs_columns: list[str]) -> ad.AnnData:
    """One member's table keyed by gene symbol, restricted to its organism's genes.

    A Visium HD member uses the 16 µm table (Task 5 step 0), or, when its store has none, the 8 µm table
    summed in 2 x 2 blocks.
    """
    from sddb.cohort import SpatialDataCohort
    from sddb.remote import elements

    table = None
    if str(row["technology"].iloc[0]) == "VisiumHD":
        table = hd_table(elements(str(row["zarr_url"].iloc[0])))
        if table == HD_FINE_TABLE and not HD_SUM_CHECKED:
            raise NoHD16Error(NO_HD_16)
    adata = SpatialDataCohort(row).to_anndata(
        table=table, gene_key=GENE_KEY, spatial="auto", obs_columns=obs_columns, max_workers=1
    )
    if table == HD_FINE_TABLE:
        adata = aggregate_bins(adata)
    return restrict_to_organism(adata, str(row["organism"].iloc[0]))


def _join_family(parts: list[ad.AnnData], obs_columns: list[str]) -> ad.AnnData:
    """Outer join of one family's members, with each gene's annotation from the first member that has it."""
    out = ad.concat(parts, join="outer")
    var = pd.concat([p.var for p in parts])
    out.var = var[~var.index.duplicated()].reindex(out.var_names)
    for col in ("uid", *obs_columns):
        if col in out.obs.columns:
            out.obs[col] = out.obs[col].astype(str).astype("category")
    return out


def load_cohort(
    spec: dict,
    release: str | None = None,
    *,
    catalog: object = None,
    families: dict[str, list[str]] | None = None,
    max_workers: int = 8,
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    """Load a cohort as one AnnData per technology family, plus the cohort table.

    The members are ``spec["query"]`` run on the catalog, plus ``spec["include_uids"]``, minus
    ``spec["exclude_uids"]``. A query without ``technology`` is limited to the family technologies. If
    ``spec["families"]`` lists family names, only those families are used (``imaging`` means
    ``imaging_cell``). ``catalog`` is used as given; otherwise ``sddb.Catalog(version=release)`` is opened.

    Each member's table is read on its own, keyed by gene symbol (the var_names of 10x tables, or
    ``var["symbol"]`` of HEST tables), with ``obsm["spatial"]`` filled from the shapes when the table has
    none. Symbols are used because some Allen MERSCOPE and MERFISH tables have no Ensembl ids. A member
    that is not redistributable, has a technology outside every family, has no paper key, or whose table
    fails to read is kept in the cohort table with ``status`` set to ``dropped: <reason>``. A table without
    gene symbols gives ``dropped: load error: no gene symbols``, a table with no gene of the member's
    organism gives ``dropped: genes from another organism`` (see ``restrict_to_organism``), any other failure
    ``dropped: load error: <ExceptionType>``. The others still load. Members are joined only within their
    family.

    Each family AnnData records the genes each member measured (after ``restrict_to_organism``) in
    ``uns["member_genes"]`` as ``{dataset_uid: [gene, ...]}``, because the outer join pads a gene a
    member lacks with zeros. The cohort table gives each loaded member's ``n_genes`` and ``n_ensembl``
    (genes with an Ensembl id in any id column).

    obs per cell or spot: ``dataset_uid``, ``paper``, ``paper_source``, ``technology``, ``tissue``,
    ``donor_id`` (when the catalog has it), ``license`` and ``license_noncommercial``.

    Raises ``ValueError`` when the kept members come from fewer than 3 papers, naming the count.
    """
    if catalog is None:
        import sddb

        catalog = sddb.Catalog(version=release)
    families = spec_families(spec, families)
    members = cohort_members(catalog, spec, families)
    status = members.pop("status")

    def table() -> pd.DataFrame:
        out = members.assign(status=status)
        return out.reindex(columns=list(TABLE_COLUMNS)).reset_index(drop=True)

    def check_papers(when: str) -> None:
        n = members.loc[status == "kept", "paper"].nunique()
        if n < MIN_PAPERS:
            raise ValueError(f"cohort {spec.get('slug', '?')!r} has {n} papers {when}; at least {MIN_PAPERS} needed")

    check_papers("after the licence filter")
    obs_columns = [c for c in OBS_COLUMNS if c in members.columns]
    kept = [i for i in members.index if status[i] == "kept"]

    def load(i: int) -> ad.AnnData | Exception:
        try:
            return _load_member(members.loc[[i]], obs_columns)
        except Exception as err:  # one bad store must not stop the cohort
            return err

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        loaded = dict(zip(kept, pool.map(load, kept), strict=True))
    for i, part in loaded.items():
        if isinstance(part, WrongOrganismError):
            status[i] = WRONG_ORGANISM
        elif isinstance(part, NoHD16Error):
            status[i] = NO_HD_16
        elif isinstance(part, ValueError) and "no usable source for gene_key" in str(part):
            status[i] = NO_SYMBOLS
        elif isinstance(part, Exception):
            status[i] = f"dropped: load error: {type(part).__name__}"
    check_papers("after loading")
    for i, part in loaded.items():
        if status[i] == "kept":
            members.loc[i, "n_genes"] = part.n_vars
            members.loc[i, "n_ensembl"] = n_ensembl(part)
    for col in ("n_genes", "n_ensembl"):
        members[col] = members.get(col, pd.Series(index=members.index, dtype=float)).astype("Int64")

    adatas: dict[str, ad.AnnData] = {}
    for fam in families:
        picked = [i for i in loaded if status[i] == "kept" and members.loc[i, "family"] == fam]
        if picked:
            adatas[fam] = _join_family([loaded[i] for i in picked], obs_columns)
            adatas[fam].uns["member_genes"] = {
                str(members.loc[i, "dataset_uid"]): loaded[i].var_names.astype(str).tolist() for i in picked
            }
    return adatas, table()


def imaging_view_allowed(cohort_table: pd.DataFrame) -> tuple[bool, str]:
    """Whether the imaging-family view may be embedded, and why.

    It is allowed only when the kept imaging members span at least 2 papers or at least 3 sections, so
    Harmony is never run on one paper with one or two sections. The reason reads like
    ``"1 paper, 2 sections"``.
    """
    kept = cohort_table[(cohort_table["status"] == "kept") & (cohort_table["family"] == "imaging_cell")]
    n_papers, n_sections = kept["paper"].nunique(), len(kept)
    reason = f"{n_papers} paper{'s' * (n_papers != 1)}, {n_sections} section{'s' * (n_sections != 1)}"
    return n_papers >= 2 or n_sections >= 3, reason


def smoke_spec(spec: dict, catalog: object, n: int) -> dict:
    """``spec`` limited to the smallest kept member of each of ``n`` papers, alternating between families."""
    members = cohort_members(catalog, spec, spec_families(spec))
    kept = members[members["status"] == "kept"].sort_values("n_obs")
    per_family = [g.drop_duplicates("paper") for _, g in kept.groupby("family")]
    chosen, papers = [], set()
    while len(chosen) < n and any(len(g) for g in per_family):
        for i, g in enumerate(per_family):
            g = g[~g["paper"].isin(papers)]
            per_family[i] = g.iloc[1:]
            if len(g) and len(chosen) < n:
                chosen.append(str(g["uid"].iloc[0]))
                papers.add(g["paper"].iloc[0])
    others = sorted(set(members["uid"].astype(str)) - set(chosen))
    return {**spec, "slug": f"{spec['slug']}-smoke", "exclude_uids": [*(spec.get("exclude_uids") or []), *others]}


def staleness(manifest: dict, catalog: object) -> dict:
    """Which datasets the primer's spec selects now that it didn't when the primer was computed, and back.

    The spec (query, include_uids, exclude_uids, families) is read from ``manifest["spec"]``, or from
    ``manifest["query"]`` for older bundles. Returns ``stale`` plus sorted ``added`` and ``removed`` uids,
    comparing ``catalog`` with the manifest's ``members_kept`` and ``members_dropped``.
    """
    spec = dict(manifest.get("spec") or {"query": manifest.get("query", {})})
    spec["families"] = spec.get("families")
    now = set(_candidates(catalog, spec, spec_families(spec))["uid"].astype(str))
    then = set(manifest.get("members_kept", [])) | set(manifest.get("members_dropped", {}))
    added, removed = sorted(now - then), sorted(then - now)
    return {"stale": bool(added or removed), "added": added, "removed": removed}
