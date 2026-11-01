"""Embedding views, the joint-view gates and the mixing statistics of a primer (Task 6).

Each technology family gets its own view on the genes measured by every member of the view (the
intersection of ``uns["member_genes"]``; the outer join of a family pads a gene a member lacks with
zeros, so the union would compare measured genes with padding). When a family's members share fewer than
200 genes, the family is split into one view per technology, named ``"<family>:<technology>"``; a
technology whose own members share fewer than 50 genes first loses members, one panel at a time, the
panel that shrinks the technology's shared genes most first, until the rest share 50 genes; a panel that
would cut the rest's shared genes by more than half is dropped too (``_drop_odd_panels``), and each drop
is recorded with its reason. A technology left with fewer than 50 shared genes
gets no view, and the reason is recorded. A joint
view on the genes shared by all families is added only when the gates pass (the mixing rule approved on
2026-11-01). Every view is embedded the same way, following the T094 spike:

1. keep the view's measured, expressed genes; a view with more than 2,000 genes keeps 2,000 highly
   variable genes (seurat_v3 on raw counts, with the section as batch, always on the CPU);
2. normalise to the median library size over all measured genes, then keep the highly variable genes and
   log-transform;
3. scale (max 10), 30 principal components, k=30 neighbours and UMAP;
4. Harmony on the section, then neighbours and UMAP again on the corrected components. A view with one
   section stays uncorrected, with a note.

Views are returned without expression (``X`` is an empty sparse matrix; ``var`` lists the genes used),
except for the marker genes: ``obsm["marker_expr"]`` keeps the log-normalised expression of every marker
and fallback gene of the marker set that the view measured, for the notebook's dotplots. The GPU path uses rapids-singlecell from step 3 on and is the
one whose numbers are published; the CPU path (scanpy and harmonypy) runs the tests.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from primers.markers import panel_genes
from primers.qc import MIN_GENES_FOR_QC, floor_loss, floor_of

SEED = 0
N_PCS, N_NEIGHBORS, N_HVG, MAX_SCALE = 30, 30, 2000, 10
MIN_SHARED_GENES, MAX_FLOOR_LOSS = 200, 0.5
CAVEAT = (
    "Well mixed is not the same as biologically right: a correction can also mix cells that differ for a "
    "real reason, such as tissue or disease."
)


def _host(x):
    """``x`` as a host numpy array: ``x.get()`` for a device (cupy) array, else ``np.asarray(x)``.

    Device arrays are recognised by ``__cuda_array_interface__``; a DataFrame also has ``.get``.
    """
    return x.get() if hasattr(x, "__cuda_array_interface__") else np.asarray(x)


def _ops(use_gpu: bool):
    """(pp, tl, harmony) for the chosen path."""
    if use_gpu:
        import rapids_singlecell as rsc

        def harmony(a, key):
            rsc.pp.harmony_integrate(a, key=key, basis="X_pca", adjusted_basis="X_pca_harmony")

        return rsc.pp, rsc.tl, harmony
    import harmonypy
    import scanpy as sc

    def harmony(a, key):
        # harmonypy 2.x returns Z_corr as (cells, components), 0.0.x as (components, cells); scanpy's wrapper
        # assumes the old layout, so orient it here
        z = np.asarray(harmonypy.run_harmony(a.obsm["X_pca"], a.obs, key, random_state=SEED, verbose=False).Z_corr)
        a.obsm["X_pca_harmony"] = z if z.shape[0] == a.n_obs else z.T

    return sc.pp, sc.tl, harmony


def _graph_and_umap(a: ad.AnnData, pp, tl, rep: str, name: str) -> None:
    """Neighbours on ``rep`` (kept as obsp[f"{name}_distances"]) and their UMAP (X_umap, or X_umap_harmony for Harmony)."""
    pp.neighbors(a, n_neighbors=N_NEIGHBORS, use_rep=rep, key_added=name, random_state=SEED)
    earlier = a.obsm.pop("X_umap", None)  # tl.umap always writes X_umap
    tl.umap(a, neighbors_key=name, random_state=SEED)
    if rep != "X_pca":
        a.obsm["X_umap_harmony"] = a.obsm.pop("X_umap")
        if earlier is not None:
            a.obsm["X_umap"] = earlier


def _measured(adata: ad.AnnData) -> list[str]:
    """The genes measured by every member of ``adata`` (``uns["member_genes"]``), in var order.

    A member missing from ``member_genes`` counts as measuring every gene of ``var``.
    """
    known = adata.uns.get("member_genes") or {}
    common = set(adata.var_names)
    for uid in adata.obs["dataset_uid"].astype(str).unique():
        if uid in known:
            common &= set(known[uid])
    return [g for g in adata.var_names if g in common]


def _common(adatas: list[ad.AnnData]) -> list[str]:
    """The genes measured by every member of every AnnData in ``adatas``, in the first one's var order."""
    if not adatas:
        return []
    common = set.intersection(*(set(_measured(a)) for a in adatas))
    return [g for g in adatas[0].var_names if g in common]


PANEL_DROP = "dropped: panel shares {n} genes with the rest of its technology"


def _drop_odd_panels(adata: ad.AnnData) -> tuple[ad.AnnData, dict[str, str], list[dict]]:
    """``adata`` without the members whose panels keep their technology under ``MIN_GENES_FOR_QC`` shared genes.

    Members with the same panel count as one unit, and the unit with the most sections (then observations)
    is the anchor, which always stays. Per technology, while its units share fewer genes than the floor, the
    unit whose removal leaves the most shared genes is dropped (on a tie, the smaller panel,
    then fewer observations). Then, latest drop first, a dropped unit is taken back when it shares the
    floor and at least half of the rest's shared genes. Last, in a technology that needed drops, a kept unit
    goes too when the rest would share more than twice the genes without it ("would reduce 500 shared
    genes to 61"). Returns the rest, ``{uid: reason}``, and per unit that costs genes against the anchor,
    ``{technology, uids, genes_without, genes_with, kept, reason}`` (dropped
    units by the halving rule are listed with ``kept: false``).
    """
    known = adata.uns.get("member_genes") or {}
    var = set(adata.var_names)
    uids = adata.obs["dataset_uid"].astype(str)
    n_obs = uids.value_counts()
    tech = adata.obs["technology"].astype(str)
    dropped, costs = {}, []

    def inter(ps):
        return set.intersection(*(set(p) for p in ps)) if ps else set()

    for t in tech.unique():
        units: dict[frozenset, list[str]] = {}
        for u in uids[(tech == t).to_numpy()].unique():
            units.setdefault(frozenset(set(known.get(u, var)) & var), []).append(u)
        size = {p: int(sum(n_obs[u] for u in us)) for p, us in units.items()}
        anchor = max(units, key=lambda p: (len(units[p]), size[p]))
        keep, out, halved = set(units), [], {}
        while len(keep) > 1 and len(inter(keep)) < MIN_GENES_FOR_QC:
            # ponytail: O(units^2) set intersections per drop; fine for hundreds of panels
            worst = max(keep - {anchor}, key=lambda p: (len(inter(keep - {p})), -len(p), -size[p]))
            keep.discard(worst)
            out.append(worst)
        if not out:
            continue
        for p in reversed(out[:]):
            shared, new = inter(keep), inter(keep) & p
            if len(new) >= MIN_GENES_FOR_QC and len(new) >= 0.5 * len(shared):
                keep.add(p)
                out.remove(p)
        while len(keep) > 1:
            shared = inter(keep)
            without = {p: inter(keep - {p}) for p in keep}
            p = max(keep - {anchor}, key=lambda p: (len(without[p]), -size[p]))
            if len(shared) >= 0.5 * len(without[p]):
                break
            keep.discard(p)
            out.append(p)
            halved[p] = (len(without[p]), len(shared))
        shared = inter(keep)
        for p in out:
            n = halved.get(p)
            reason = (
                f"dropped: panel would reduce {n[0]} shared genes to {n[1]}"
                if n
                else PANEL_DROP.format(n=len(p & shared))
            )
            dropped.update(dict.fromkeys(units[p], reason))
            if n:
                costs.append(
                    {
                        "technology": t,
                        "uids": units[p],
                        "genes_without": n[0],
                        "genes_with": n[1],
                        "kept": False,
                        "reason": reason,
                    }
                )
        for p in keep - {anchor}:
            alone = inter(keep - {p})
            if len(alone) > len(shared):
                costs.append(
                    {
                        "technology": t,
                        "uids": units[p],
                        "genes_without": len(alone),
                        "genes_with": len(shared),
                        "kept": True,
                        "reason": f"kept: shares {len(shared)} of {len(alone)}",
                    }
                )
    if not dropped:
        return adata, {}, costs
    return adata[~uids.isin(dropped).to_numpy()], dropped, costs


def _family_views(family: str, adata: ad.AnnData) -> tuple[list[tuple[str, ad.AnnData, list[str], str]], dict]:
    """The views of one family as (name, observations, genes, note), and the technologies left without one.

    The family is one view on its members' shared genes, or, when they share fewer than 200 genes, one
    view per technology on that technology's shared genes. A technology with fewer than 50 shared genes
    gets no view.
    """
    genes = _measured(adata)
    if len(genes) >= MIN_SHARED_GENES:
        return [(family, adata, genes, "")], {}
    tech = adata.obs["technology"].astype(str)
    if tech.nunique() == 1:  # nothing to split; the family is one technology
        if len(genes) < MIN_GENES_FOR_QC:
            return [], {
                family: f"its members share only {len(genes)} measured genes (under {MIN_GENES_FOR_QC}), so no view"
            }
        return [(family, adata, genes, f"only {len(genes)} shared genes (under {MIN_SHARED_GENES})")], {}
    per_tech = {t: _measured(adata[(tech == t).to_numpy()]) for t in tech.unique()}
    panels = " ∩ ".join(f"{t} {len(g)}" for t, g in sorted(per_tech.items(), key=lambda x: len(x[1])))
    note = f"{panels} = {len(genes)} shared genes (under {MIN_SHARED_GENES}), so one view per technology"
    views, skipped = [], {}
    for t, g in per_tech.items():
        name = f"{family}:{t}"
        if len(g) < MIN_GENES_FOR_QC:
            skipped[name] = f"its members share only {len(g)} measured genes (under {MIN_GENES_FOR_QC}), so no view"
        else:
            views.append((name, adata[(tech == t).to_numpy()], g, note))
    return views, skipped


def _embed(
    adata: ad.AnnData,
    genes: list[str],
    batch_key: str,
    use_gpu: bool,
    family: str,
    note: str,
    markers: set[str] = frozenset(),
) -> ad.AnnData:
    """One view on ``genes``: PCA, UMAP and, with two or more sections, their Harmony versions; X is emptied.

    ``obsm["marker_expr"]`` keeps log1p(counts / size factor) of the genes in ``genes`` whose upper-cased
    symbol is in ``markers``, as a DataFrame; a measured gene with no counts in the view is a column of zeros.
    """
    import scanpy as sc

    pp, tl, harmony = _ops(use_gpu)
    expressed = np.asarray(adata.X.sum(axis=0)).ravel() > 0  # the GPU sparse PCA refuses unexpressed genes
    a = adata[:, np.isin(adata.var_names, genes) & expressed].copy()
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    n_sections = a.obs[batch_key].nunique()
    hvg = a.n_vars > N_HVG
    if hvg:  # on raw counts, on the CPU (rapids-singlecell's seurat_v3 needs the counts on the GPU)
        try:
            sc.pp.highly_variable_genes(
                a, n_top_genes=N_HVG, flavor="seurat_v3", batch_key=batch_key if n_sections > 1 else None
            )
            how = "section as batch" if n_sections > 1 else "one section"
        except ValueError:  # the per-section loess fit can fail on very small sections
            sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat_v3")
            how = "without batch, because the per-section fit failed"
        rule = f"{N_HVG} highly variable genes of the {a.n_vars} measured, expressed genes (seurat_v3 on raw counts, {how})"
    else:
        rule = f"all {a.n_vars} measured, expressed genes of the view"
    totals = np.asarray(a.X.sum(axis=1)).ravel()
    a.obs["size_factor"] = totals / np.median(totals[totals > 0])
    sc.pp.normalize_total(a)  # to the median library size over all measured genes
    marker_cols = [g for g in genes if g.upper() in markers]
    present = [g for g in marker_cols if g in a.var_names]
    marker_expr = pd.DataFrame(
        np.log1p(sp.csr_matrix(a[:, present].X).toarray()), index=a.obs_names, columns=present
    ).reindex(columns=marker_cols, fill_value=0.0)
    if hvg:
        a = a[:, a.var["highly_variable"].to_numpy()].copy()
    sc.pp.log1p(a)
    if use_gpu:
        import rapids_singlecell as rsc

        rsc.get.anndata_to_GPU(a)
    pp.scale(a, max_value=MAX_SCALE)
    pp.pca(a, n_comps=min(N_PCS, a.n_vars - 1, a.n_obs - 1), random_state=SEED)
    a.X = sp.csr_matrix(a.shape, dtype=np.float32)  # free the scaled matrix before the graphs
    _graph_and_umap(a, pp, tl, "X_pca", "nn_pca")
    corrected = n_sections > 1
    if corrected:
        harmony(a, batch_key)
        a.obsm["X_pca_harmony"] = _host(a.obsm["X_pca_harmony"]).astype(np.float32)
        _graph_and_umap(a, pp, tl, "X_pca_harmony", "nn_harmony")
    for key in list(a.obsm):
        a.obsm[key] = _host(a.obsm[key])
    a.obsm["marker_expr"] = marker_expr
    for key in list(a.obsp):
        if hasattr(a.obsp[key], "get"):
            a.obsp[key] = a.obsp[key].get()
    notes = [note] if note else []
    if not corrected:
        notes.append("one section, so no batch correction; the embedding is uncorrected")
    a.uns["primer"] = {
        "corrected": corrected,
        "path": "gpu" if use_gpu else "cpu",
        "family": family,
        "genes": {"measured": len(genes), "used": int(a.n_vars), "rule": rule, "size_factors": "all measured genes"},
        "note": "; ".join(notes),
        "n_sections": int(n_sections),
        "n_obs": int(a.n_obs),
    }
    return a


def _compartment_genes(marker_set: dict) -> dict[str, set[str]]:
    """Per compartment, the upper-cased genes that measure it: its markers and every fallback gene for it.

    ``not_measured`` fallbacks never count.
    """
    out = {c: {str(g).upper() for g in genes} for c, genes in marker_set["compartments"].items()}
    for entries in (marker_set.get("fallback") or {}).values():
        for c, fb in (entries or {}).items():
            if c in out and isinstance(fb, dict) and not fb.get("not_measured"):
                out[c] |= {str(g).upper() for g in fb.get("genes") or []}
    return out


def _measures(candidates: dict[str, set[str]], genes: list[str]) -> set[str]:
    upper = {g.upper() for g in panel_genes(genes)}
    return {c for c, markers in candidates.items() if markers & upper}


def _floor_text(floor: float | dict[str, float], tech: str) -> str:
    return f"{floor[tech] if isinstance(floor, dict) else floor:g}"


def joint_gates(adatas: dict[str, ad.AnnData], marker_set: dict, floor: float | dict[str, float]) -> dict:
    """The gates of the joint view, with every dropped technology and its reason, in order.

    Shared genes are those measured by every member (``uns["member_genes"]``). Technologies are dropped
    in this order (the Lead's ruling of 2026-11-01), each with a logged reason:

    1. a technology whose own members share fewer than 50 genes;
    2. then, repeatedly on the genes shared by the remaining technologies: gate 3 drops a technology that
       loses more than half its observations below its ``floor`` (one number, or one per technology);
       if no technology is dropped and fewer than 200 genes are shared, the technology with the smallest
       panel is dropped. This stops when 200 genes are shared or fewer than two families remain.

    Then gate 1 (at least 200 shared genes) and gate 2 (a compartment is expected when every remaining
    family can measure it by a marker or a fallback gene of ``marker_set``; every expected compartment
    needs one among the shared genes; the gate fails when none is expected) are evaluated. ``joint`` is
    True when both pass and at least two families remain. ``drops`` lists each dropped technology as
    {technology, step, reason}; ``gate3["excluded"]`` maps every dropped technology to its reason.
    """
    tech_rows = {
        (f, t): a[(a.obs["technology"].astype(str) == t).to_numpy()]
        for f, a in adatas.items()
        for t in a.obs["technology"].astype(str).unique()
    }
    panel = {t: len(_measured(a)) for (_, t), a in tech_rows.items()}
    drops: list[dict] = []
    for (_, t), n in sorted(((k, panel[k[1]]) for k in tech_rows), key=lambda x: x[1]):
        if n < MIN_GENES_FOR_QC:
            drops.append(
                {
                    "technology": t,
                    "step": "panel",
                    "reason": f"its members share only {n} genes (under {MIN_GENES_FOR_QC})",
                }
            )
    remaining = {k: a for k, a in tech_rows.items() if k[1] not in {d["technology"] for d in drops}}
    loss: dict[str, float] = {}

    def families(rows: dict) -> set[str]:
        return {f for f, _ in rows}

    while len(families(remaining)) >= 2:
        shared = _common(list(remaining.values()))
        step_loss = {t: floor_loss(a, floor, genes=shared)[t] for (_, t), a in remaining.items()}
        loss.update(step_loss)
        lost = [t for t, share in step_loss.items() if share > MAX_FLOOR_LOSS]
        if lost:
            for t in lost:
                drops.append(
                    {
                        "technology": t,
                        "step": "gate3",
                        "reason": f"loses {step_loss[t]:.0%} of its observations below {_floor_text(floor, t)} counts on the "
                        f"{len(shared)} shared genes (limit {MAX_FLOOR_LOSS:.0%})",
                    }
                )
            remaining = {k: a for k, a in remaining.items() if k[1] not in lost}
            continue
        if len(shared) >= MIN_SHARED_GENES:
            break
        (f_small, t_small) = min(remaining, key=lambda k: panel[k[1]])
        drops.append(
            {
                "technology": t_small,
                "step": "gate1",
                "reason": f"smallest panel ({panel[t_small]} genes) while only {len(shared)} genes were shared "
                f"(at least {MIN_SHARED_GENES} needed)",
            }
        )
        remaining = {k: a for k, a in remaining.items() if k != (f_small, t_small)}
    shared = _common(list(remaining.values()))
    excluded = {d["technology"]: d["reason"] for d in drops}
    kept_families = families(remaining)
    gate3 = {"loss": loss, "excluded": excluded, "floor": floor, "n_shared": len(shared)}
    gate1 = {
        "passed": len(shared) >= MIN_SHARED_GENES and len(kept_families) >= 2,
        "n_shared": len(shared),
        "n_shared_all": len(_common(list(tech_rows.values()))),
        "min": MIN_SHARED_GENES,
    }
    by_family: dict[str, list[ad.AnnData]] = {}
    for (f, _), a in remaining.items():
        by_family.setdefault(f, []).append(a)
    candidates = _compartment_genes(marker_set)
    can = [_measures(candidates, _common(parts)) for parts in by_family.values()]
    expected = [c for c in candidates if can and all(c in m for m in can)]
    found = _measures(candidates, shared)
    missing = [c for c in expected if c not in found]
    gate2 = {"passed": bool(expected) and not missing, "expected": expected, "missing": missing}
    joint = gate1["passed"] and gate2["passed"]
    if len(kept_families) < 2:
        reason = "fewer than two families keep a technology"
    elif not gate1["passed"]:
        reason = f"only {len(shared)} shared genes (at least {MIN_SHARED_GENES} needed)"
    elif not expected:
        reason = "no compartment measurable by every family"
    elif missing:
        reason = f"no marker among the shared genes for {', '.join(missing)}"
    else:
        reason = ""
    return {
        "joint": joint,
        "reason": reason,
        "shared_genes": shared,
        "drops": drops,
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
    }


def embed_views(
    adatas: dict[str, ad.AnnData],
    marker_set: dict,
    floor: float | dict[str, float],
    batch_key: str = "dataset_uid",
    use_gpu: bool = True,
) -> dict[str, ad.AnnData]:
    """The views of a cohort (``adatas`` from ``load_cohort`` after QC), plus "joint" when the gates pass.

    ``marker_set`` is one set of ``markers.yaml`` (``compartments`` and optional ``fallback``). A family
    is one view, or one ``"<family>:<technology>"`` view per technology when its members share fewer than
    200 genes. Each view has ``X_pca``, ``X_umap`` and, with two or more sections, ``X_pca_harmony`` and
    ``X_umap_harmony``, obs ``size_factor``, and ``uns["primer"]`` with ``corrected``, ``path``,
    ``family``, ``genes`` (``measured``, ``used``, ``rule``, ``size_factors``), ``note``,
    ``views_skipped`` (technologies without a view and why), ``members_dropped_panel`` (members dropped
    because their panel kept their technology under 50 shared genes, see ``_drop_odd_panels``),
    ``panel_costs`` (panels that lower the view's shared genes, kept or dropped, see ``_drop_odd_panels``) and, for a cohort with two or more families,
    ``gates`` (``shared_genes`` left out). The joint view also records ``excluded`` technologies and their
    reasons. ``obsm["marker_expr"]`` is the log-normalised expression (obs x genes, a DataFrame) of every
    marker and fallback gene of ``marker_set`` that the view measured, matched without regard to case.
    """
    views, skipped, dropped, costs = {}, {}, {}, []
    markers = set().union(*_compartment_genes(marker_set).values())
    adatas = dict(adatas)
    for fam in adatas:  # before the family and joint gene sets, so an odd panel can't shrink them
        adatas[fam], d, c = _drop_odd_panels(adatas[fam])
        dropped.update(d)
        costs += c
    for fam, a in adatas.items():
        parts, skip = _family_views(fam, a)
        skipped.update(skip)
        for name, sub, genes, note in parts:
            views[name] = _embed(sub, genes, batch_key, use_gpu, family=fam, note=note, markers=markers)
    if not views:
        raise ValueError(f"no view left: {skipped}")
    gates = None
    if len(adatas) >= 2:
        gates = joint_gates(adatas, marker_set, floor)
        shared, excluded = gates.pop("shared_genes"), gates["gate3"]["excluded"]
        if gates["joint"]:
            parts = [a[~a.obs["technology"].astype(str).isin(excluded).to_numpy(), shared] for a in adatas.values()]
            joint = ad.concat(parts, join="inner")
            on_shared = np.asarray(joint.X.sum(axis=1)).ravel()
            views["joint"] = _embed(
                joint[on_shared >= floor_of(joint, floor)],
                shared,
                batch_key,
                use_gpu,
                family="joint",
                note="",
                markers=markers,
            )
            views["joint"].uns["primer"]["excluded"] = excluded
    for v in views.values():
        v.uns["primer"]["views_skipped"] = skipped
        v.uns["primer"]["members_dropped_panel"] = dropped
        techs = set(v.obs["technology"].astype(str))
        v.uns["primer"]["panel_costs"] = [c for c in costs if c["technology"] in techs]
        v.uns["primer"]["floor"] = floor
        if gates is not None:
            v.uns["primer"]["gates"] = gates
    return views


def _knn(adata: ad.AnnData, basis: str, k: int) -> sp.csr_matrix:
    """The kNN graph of ``basis`` as CSR: the stored neighbour graph when there is one, else a new one."""
    key = {"X_pca": "nn_pca", "X_pca_harmony": "nn_harmony"}.get(basis)
    if key and f"{key}_distances" in adata.obsp:
        graph = adata.obsp[f"{key}_distances"]
        return sp.csr_matrix(graph.get() if hasattr(graph, "get") else graph)
    from sklearn.neighbors import NearestNeighbors

    return sp.csr_matrix(
        NearestNeighbors(n_neighbors=min(k, adata.n_obs - 1))
        .fit(_host(adata.obsm[basis]))
        .kneighbors_graph(mode="connectivity")
    )


def _neighbour_stats(graph: sp.csr_matrix, codes: np.ndarray, n_batches: int) -> tuple[np.ndarray, np.ndarray]:
    """Per observation, the normalised neighbour batch entropy and the other-batch share.

    Counts come from the graph's CSR structure, one (row, batch) pair per stored edge, so memory stays
    O(n * k). Rows without neighbours get 0 for both; the caller leaves them out.
    """
    n = graph.shape[0]
    deg = np.diff(graph.indptr)
    rows = np.repeat(np.arange(n, dtype=np.int64), deg)
    pairs, counts = np.unique(rows * n_batches + codes[graph.indices], return_counts=True)
    r, c = pairs // n_batches, pairs % n_batches
    p = counts / deg[r]
    h = np.bincount(r, weights=-p * np.log(p), minlength=n)
    norm = np.log(np.minimum(deg, n_batches))
    entropy = np.divide(h, norm, out=np.zeros(n), where=norm > 0)
    same = np.bincount(r, weights=np.where(c == codes[r], counts, 0), minlength=n)
    other = 1.0 - np.divide(same, deg, out=np.ones(n), where=deg > 0)
    return entropy, other


def mixing_stats(adata: ad.AnnData, batch_key: str, basis: str, k: int = N_NEIGHBORS) -> dict:
    """Two named statistics of how well batches mix among each observation's nearest neighbours.

    - ``entropy``: the entropy of the batch labels among an observation's neighbours, divided by
      log(min(k, number of batches)), its largest possible value, so 0 means all neighbours share one
      batch and 1 means the batches are as evenly spread as k neighbours allow.
    - ``other_share``: the share of an observation's neighbours that come from another batch.

    The neighbours are the stored graph of ``basis`` (``k`` reports the neighbours per observation it
    holds), else a new k=30 graph. Each statistic gives the mean over observations, the mean per batch,
    and a permutation reference: the same statistic with the batch labels shuffled over the same graph
    (seed ``SEED``), as ``random`` and ``random_per_batch``. ``random_limit`` is the value expected from
    the batch proportions alone. Observations without neighbours are left out of the means and counted in
    ``n_without_neighbours``. With one batch both statistics are undefined, and their values are None,
    with a ``note``. ``caveat`` says that good mixing is not the same as a right biology.
    """
    batches = adata.obs[batch_key].astype(object).fillna("(missing)").astype(str)  # a missing label is one batch
    labels, codes = np.unique(batches.to_numpy(), return_inverse=True)
    n_b = len(labels)
    graph = _knn(adata, basis, k)
    deg = np.diff(graph.indptr)
    has = deg > 0
    k_used = int(np.median(deg[has])) if has.any() else 0
    definitions = {
        "entropy": f"How evenly the {batch_key} labels are spread among each observation's {k_used} nearest "
        f"neighbours, divided by log(min({k_used}, number of labels)): 0 when all neighbours share one label, "
        "1 when the labels are as evenly spread as the neighbours allow.",
        "other_share": f"The share of each observation's {k_used} nearest neighbours that carry another {batch_key}.",
    }
    out = {
        "batch_key": batch_key,
        "basis": basis,
        "k": k_used,
        "n_batches": int(n_b),
        "n_without_neighbours": int((~has).sum()),
        "note": "",
        "caveat": CAVEAT,
    }
    if n_b < 2:
        out["note"] = f"one {batch_key}, so neither statistic is defined"
        for name, d in definitions.items():
            out[name] = dict.fromkeys(("mean", "per_batch", "random", "random_per_batch", "random_limit"))
            out[name]["definition"] = d
        return out
    shuffled = np.random.default_rng(SEED).permutation(codes)
    observed = dict(zip(("entropy", "other_share"), _neighbour_stats(graph, codes, n_b), strict=True))
    random = dict(zip(("entropy", "other_share"), _neighbour_stats(graph, shuffled, n_b), strict=True))
    prop = np.bincount(codes, minlength=n_b) / len(codes)
    limits = {
        "entropy": float(-(prop * np.log(prop)).sum() / np.log(min(k_used, n_b))) if min(k_used, n_b) > 1 else None,
        "other_share": 1.0 - float((prop**2).sum()),
    }

    def per_batch(values: np.ndarray, by: np.ndarray) -> dict[str, float]:
        means = pd.Series(values[has]).groupby(by[has]).mean()
        return {str(labels[i]): float(v) for i, v in means.items()}

    for name, d in definitions.items():
        out[name] = {
            "mean": float(observed[name][has].mean()),
            "per_batch": per_batch(observed[name], codes),
            "random": float(random[name][has].mean()),
            "random_per_batch": per_batch(random[name], shuffled),
            "random_limit": limits[name],
            "definition": d,
        }
    return out
