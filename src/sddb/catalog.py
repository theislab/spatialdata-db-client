"""Catalog fetch, validation, filtering and search."""

from __future__ import annotations

import operator
import os
import re
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from sddb import _catalog_schema as schema
from sddb._cache import _file_sha256, _local_path, fetch_catalog
from sddb.cohort import SpatialDataCohort
from sddb.dataset import Dataset, Source

if TYPE_CHECKING:
    from sddb.features import FeatureIndex
    from sddb.genes import GeneIndex

_ROLLING_SUFFIX = "/catalog-latest/catalog.parquet"
_RELEASE_BASE = "https://github.com/theislab/spatialdata-db-client/releases/download"
DEFAULT_CATALOG_URL = _RELEASE_BASE + _ROLLING_SUFFIX
SITE_URL = "https://spatialdata-db-web.pages.dev"  # the DB website; dataset pages are SITE_URL/datasets/<slug>/
_FUZZY_CUTOFF = 88
_RANGE_OPS = {"gte": operator.ge, "gt": operator.gt, "lte": operator.le, "lt": operator.lt}
_SUGGEST_CUTOFF = 70


def _split_facets(facets: Mapping[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str, Any]]]:
    """Separate plain equality facets from ``col__op`` range predicates (op in _RANGE_OPS)."""
    equality: dict[str, Any] = {}
    ranges: list[tuple[str, str, Any]] = []
    for key, val in facets.items():
        base, _, op = key.rpartition("__")
        if base and op in _RANGE_OPS:
            ranges.append((base, op, val))
        else:
            equality[key] = val
    return equality, ranges


def _token_matches(token: str, value: str) -> bool:
    """Match a lowercase token against a facet value: whole word first, fuzzy/substring for longer tokens."""
    low = value.lower()
    words = re.findall(r"\w+", low)
    if token == low or token in words:
        return True
    if len(token) >= 5 and token in low:
        return True
    return len(token) >= 4 and any(fuzz.ratio(token, w) >= _FUZZY_CUTOFF for w in words)


def _is_list_col(series: pd.Series) -> bool:
    """True if ``series`` holds list/array cells (a list-valued facet such as ``modalities``)."""
    return any(not isinstance(v, str) and hasattr(v, "__iter__") for v in series.dropna())


def _distinct(series: pd.Series) -> list[str]:
    """Sorted distinct non-null values; list-valued columns are exploded element-wise."""
    if _is_list_col(series):
        return sorted({str(e) for v in series.dropna() for e in v})
    return sorted(str(v) for v in series.dropna().unique())


def _facet_isin(series: pd.Series, values: list[Any]) -> pd.Series:
    """Row mask for facet membership; for list-valued columns, any element in ``values`` matches."""
    if _is_list_col(series):
        want = set(values)
        return series.map(
            lambda v: not (v is None or (not hasattr(v, "__iter__") and pd.isna(v))) and any(e in want for e in v)
        )
    return series.isin(values)


def _norm(value: Any) -> str:
    """Alias key for a facet value: lowercase, with spaces, hyphens and underscores removed."""
    return re.sub(r"[\s_-]+", "", str(value).lower())


def closest_value(value: str, known: Iterable[str]) -> str | None:
    """The known value closest to ``value`` (rapidfuzz ``ratio`` on :func:`_norm` keys, cutoff 70), or None."""
    best = process.extractOne(value, list(known), processor=_norm, scorer=fuzz.ratio, score_cutoff=_SUGGEST_CUTOFF)
    return best[0] if best else None


def _is_bool_facet(col: str) -> bool:
    return schema.CATALOG_COLUMNS.get(col) == "boolean"


def resolve_facet_values(df: pd.DataFrame, col: str, values: list[Any]) -> list[Any]:
    """Map requested values of facet ``col`` onto values present in ``df[col]``.

    A value present as given is kept. Otherwise it resolves through aliases: values that are equal
    after :func:`_norm` (case, spaces, hyphens and underscores ignored), and, for ``technology``, the
    catalog's own ``assay`` display values (``"10x Xenium"`` resolves to ``"Xenium"``; an assay shared
    by several technologies resolves to all of them). Boolean facets take ``True`` or ``False`` only.

    Raises
    ------
    ValueError
        If a value does not resolve, with the closest known value as a "did you mean" hint.
    """
    if _is_bool_facet(col):
        bad = [v for v in values if not isinstance(v, (bool, np.bool_))]
        if bad:
            raise ValueError(f"facet {col!r} takes True or False, got {bad[0]!r}")
        return values
    known = _distinct(df[col])
    own: dict[str, set[str]] = {}
    for v in known:
        own.setdefault(_norm(v), set()).add(v)
    labels: dict[str, set[str]] = {}
    if col == "technology" and "assay" in df.columns:
        pairs = df[["technology", "assay"]].dropna().drop_duplicates()
        for tech, assay in pairs.itertuples(index=False):
            labels.setdefault(_norm(assay), set()).add(str(tech))
    known_set = set(known)
    out: list[Any] = []
    for v in values:
        s = str(v)
        hit = {s} if s in known_set else own.get(_norm(s)) or labels.get(_norm(s))
        if not hit:
            choices = [*known, *sorted({str(a) for a in df["assay"].dropna()})] if labels else known
            best = closest_value(s, choices)
            hint = f"did you mean {best!r}?" if best else f"list the values with facets({col!r})"
            raise ValueError(f"unknown {col} value {s!r}; {hint}")
        out.extend(sorted(hit))
    return out


def _derive_source(study_id: pd.Series) -> pd.Series:
    """The lowercased ``study_id`` prefix before ``":"`` (``"geo:GSM1"`` -> ``"geo"``); null without one."""
    return study_id.astype("string").str.extract(r"^([^:]+):", expand=False).str.lower().astype("string")


def facet_values(df: pd.DataFrame, field: str | None = None) -> list[str]:
    """Facet columns present in ``df``, or the sorted distinct non-null values of ``field``.

    Raises ``ValueError`` if ``field`` is not a known facet or is absent from ``df``.
    """
    if field is None:
        return [c for c in schema.FACETS if c in df.columns]
    if field not in schema.FACETS or field not in df.columns:
        raise ValueError(f"unknown facet {field!r}; valid facets: {list(schema.FACETS)}")
    return _distinct(df[field])


class Catalog:
    """The dataset catalog: fetched, cached and validated on construction."""

    def __init__(
        self,
        url: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        version: str | None = None,
        refresh: bool = False,
    ) -> None:
        """Fetch + cache + validate the catalog.

        Parameters
        ----------
        url
            Catalog parquet URL or path; falls back to the ``SDDB_CATALOG_URL`` env var, then ``DEFAULT_CATALOG_URL``.
        cache_dir
            Cache directory override.
        version
            Dated snapshot (e.g. ``"2026-10-16"``); selects the ``catalog-<version>`` release in place of ``catalog-latest``.
            Ignored if an explicit ``url`` is given or the resolved URL is not the default rolling-release URL.
        refresh
            Bypass the cache and re-download.
        """
        version_applied = False
        if url is None:
            url = os.environ.get("SDDB_CATALOG_URL") or DEFAULT_CATALOG_URL
            if version is not None and url.endswith(_ROLLING_SUFFIX):
                url = url.removesuffix(_ROLLING_SUFFIX) + f"/catalog-{version}/catalog.parquet"
                version_applied = True
        if version is not None and not version_applied:
            warnings.warn(f"version={version!r} is ignored for catalog URL {url!r}", UserWarning, stacklevel=2)
        local = _local_path(url)
        if local is not None and not url.startswith("file://"):
            url = str(local.resolve())  # stable cache key regardless of cwd
        self.url = url
        self.version = version
        self._cache_dir = cache_dir
        self._refresh = refresh
        self._genes: GeneIndex | None = None
        self._features: FeatureIndex | None = None
        self._path = fetch_catalog(url, cache_dir=cache_dir, refresh=refresh)
        self._sha256 = _file_sha256(self._path)  # recorded in every cohort's Source, checked by verify()
        # The dated release behind this file: catalog-latest is an alias of the newest one, whose date is the
        # parquet's sddb_generated_at. Only the default release URLs have a release.
        self.release = version if version_applied else None
        if url.endswith(_ROLLING_SUFFIX):
            self.release = (self.generated_at or "")[:10] or None
        try:
            df = pd.read_parquet(self._path)
        except Exception as err:
            raise ValueError(f"catalog at {url} is missing or corrupt: {err}") from err
        for col, dtype in schema.CATALOG_COLUMNS.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(pd.api.types.pandas_dtype(dtype))
                except (TypeError, ValueError):
                    pass  # left for validate() to report
        try:
            schema.validate(df)
        except schema.SchemaError as err:
            raise schema.SchemaError(f"catalog at {url} is invalid: {err}") from err
        if "study_id" in df.columns and "source" not in df.columns:
            df["source"] = _derive_source(df["study_id"])
        self._df = df

    @classmethod
    def from_file(cls, path: str | Path, *, cache_dir: str | Path | None = None) -> Catalog:
        """Load a catalog from a local ``catalog.parquet`` (absolute or relative path, or ``file://`` URL).

        Parameters
        ----------
        path
            Local parquet path; validated against the catalog schema like any other catalog.
        cache_dir
            Cache directory override.
        """
        return cls(str(path), cache_dir=cache_dir)

    @property
    def generated_at(self) -> str | None:
        """Generation timestamp from the parquet metadata, if present."""
        meta = pq.read_metadata(self._path).metadata or {}
        value = meta.get(b"sddb_generated_at")
        return value.decode() if value else None

    def _source(self) -> Source:
        return Source(self.url, self._cache_dir, self.version, self._refresh, self._sha256, self.release)

    @property
    def genes(self) -> GeneIndex:
        """Cross-dataset gene index bound to this catalog (loaded lazily)."""
        if self._genes is None:
            from sddb.genes import GeneIndex

            self._genes = GeneIndex(self)
        return self._genes

    @property
    def features(self) -> FeatureIndex:
        """Cross-dataset multi-modal feature index bound to this catalog (loaded lazily)."""
        if self._features is None:
            from sddb.features import FeatureIndex

            self._features = FeatureIndex(self)
        return self._features

    def to_df(self) -> pd.DataFrame:
        """Return the full catalog as a DataFrame copy."""
        return self._df.copy()

    def facets(self, name: str | None = None) -> list[str]:
        """List the facet columns, or the values of one facet.

        These are the names and values that :meth:`query` accepts, and the list that the "did you mean" hint
        of an unknown value points to. It is the same list as ``sddb-cli facets`` and the MCP ``facets`` tool.

        Parameters
        ----------
        name
            A facet column such as ``"organism"``. ``None`` lists the facet columns in this catalog.

        Returns
        -------
        list[str]
            The facet columns, or the sorted distinct non-null values of ``name``. A list-valued facet
            (``modalities``) gives each element once.

        Raises
        ------
        ValueError
            If ``name`` is not a facet of this catalog.
        """
        return facet_values(self._df, name)

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"<Catalog: {len(self)} datasets>"

    def query(
        self,
        *,
        validation: str | None = "pass",
        license_set: bool | None = None,
        noncommercial: bool | None = None,
        redistributable: bool | None = None,
        search: str | None = None,
        expressing: str | list[str] | None = None,
        mode: str = "all",
        min_fraction: float | None = None,
        **facets: str | float | list[str] | tuple[str, ...] | set[str],
    ) -> SpatialDataCohort:
        """Filter the catalog.

        To start from known uids or study ids, use :meth:`Catalog.cohort`.

        Parameters
        ----------
        validation
            Keep rows with this ``validation_status`` (default ``"pass"``); ``None`` disables.
            In the current catalog every dataset passes validation, so this filter changes nothing yet.
            Filter on ``tier`` (``silver``, ``bronze``) for quality.
        license_set
            If True, keep only rows with a known license (``license_unknown`` is False); False behaves like None.
        noncommercial
            If False, keep only data cleared for commercial use: ``license_noncommercial`` is False and the
            license is known (``license_unknown`` is False), the same set the website counts as "Commercial
            use OK". Rows with an unknown license are excluded because their terms are not known. If True,
            keep only NonCommercial data; ``None`` disables.
        redistributable
            If True, keep only data whose license allows redistribution: ``license_redistributable`` is True
            and the license is known (``license_unknown`` is False). If False, keep the rest, which includes
            every row with an unknown license. ``None`` disables. Every dataset is served. The default ``None`` applies no license filter, so rows with an
            unknown license stay in the result; check the license before you reuse data.
        search
            Free text, as in :meth:`search`; the result is intersected with the other filters and
            carries the ``matched`` facet values. With ``validation=None`` (or ``"all"``), ``search``
            still imposes its own ``validation_status == "pass"`` filter unless ``validation_status``
            was itself matched, so failed rows do not survive a ``search``.
        expressing
            Gene symbol or Ensembl id, or a list of them; keep datasets where they are expressed (see
            ``genes.where_expressed``). The first use downloads the gene index (~900 MB, cached).
        mode
            With several ``expressing`` genes: ``"all"`` (default, AND) or ``"any"`` (union).
        min_fraction
            Minimum fraction of expressing cells for ``expressing``; requires ``expressing``.
        **facets
            Facet column -> value (equality) or list/tuple/set of values (isin). ``license`` and
            ``license_noncommercial`` are facets, so you can also filter by exact license id.
            String facets compare a value against that column of the whole catalog first. A value that is
            not there resolves through aliases: case, spaces, hyphens and underscores are ignored, and a
            ``technology`` value may be an ``assay`` display value (``technology="10x Xenium"`` keeps
            ``"Xenium"``). A value that still does not resolve raises ``ValueError`` with a "did you mean"
            hint. A known value that matches nothing after the other filters gives an empty cohort.
            ``hne_image`` and ``if_image`` take True or False. Null means unknown and matches neither.
            ``source`` is derived on load as the lowercased ``study_id`` prefix before ``":"``; rows
            without a ``study_id`` have a null source. In the 2026-10-28 catalog (6087 rows) it is
            ``geo`` 4113, ``hest`` 1070, ``10x`` 175, ``abc`` 115, ``bia`` 4, ``metaspace`` 3,
            ``bruker`` 3, ``doi`` 2, ``figshare`` 2, and null 600.
            ``<col>__gte`` / ``__gt`` / ``__lte`` / ``__lt`` apply a numeric range on a column in
            ``RANGE_COLUMNS`` (e.g. ``n_obs__gte=50_000``); missing values never match.

        Raises
        ------
        ValueError
            If a keyword is not a facet or range column, a facet column is absent from the loaded catalog,
            a facet value is not in the catalog and has no alias, a boolean facet gets a non-bool value,
            a range value is not numeric, or ``min_fraction`` is given without ``expressing``.
        """
        if mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', got {mode!r}")
        if min_fraction is not None and expressing is None:
            raise ValueError("min_fraction requires expressing=<gene>")
        equality, ranges = _split_facets(facets)
        bad = [k for k in equality if k not in schema.FACETS]
        if bad:
            raise ValueError(f"unknown facet(s) {bad}; valid facets: {list(schema.FACETS)}")
        df = self._df
        unknown_range = [c for c, _, _ in ranges if c not in schema.RANGE_COLUMNS]
        if unknown_range:
            raise ValueError(f"unknown range column(s) {unknown_range}; valid: {list(schema.RANGE_COLUMNS)}")
        absent_range = [c for c, _, _ in ranges if c not in df.columns]
        if absent_range:
            raise ValueError(f"range column(s) {absent_range} not present in this catalog")
        mask = pd.Series(True, index=df.index)
        if validation is not None:
            mask &= df["validation_status"] == validation
        if license_set and "license_unknown" in df.columns:
            mask &= df["license_unknown"].fillna(True) == False  # noqa: E712
        if noncommercial is not None and "license_noncommercial" in df.columns:
            mask &= df["license_noncommercial"].fillna(False) == noncommercial
        if noncommercial is False and "license_unknown" in df.columns:
            mask &= df["license_unknown"].fillna(True) == False  # noqa: E712
        if redistributable is not None and "license_redistributable" in df.columns:
            ok = df["license_redistributable"].fillna(False).astype(bool)
            if "license_unknown" in df.columns:
                ok &= df["license_unknown"].fillna(True) == False  # noqa: E712
            mask &= ok if redistributable else ~ok
        for col, val in equality.items():
            if col not in df.columns:
                raise ValueError(f"facet column {col!r} is not present in this catalog (columns: {list(df.columns)})")
            values = resolve_facet_values(df, col, list(val) if isinstance(val, (list, tuple, set)) else [val])
            mask &= _facet_isin(df[col], values)
        for col, op, val in ranges:
            if isinstance(val, (list, tuple, set)):
                raise ValueError(f"range value for {col}__{op} must be a scalar number, got {val!r}")
            try:
                mask &= _RANGE_OPS[op](df[col], val)
            except TypeError as err:
                raise ValueError(f"range value for {col}__{op} must be numeric, got {val!r}") from err
        keep = mask.fillna(False).astype(bool)
        # search and expressing each constrain the surviving uids; intersect into one mask, build SpatialDataCohort once.
        matched: dict[str, list[str]] | None = None
        if search is not None and keep.any():
            hits = self.search(search)
            matched = hits.matched or None
            keep &= df["uid"].isin(hits.to_df()["uid"])
        if expressing is not None and keep.any():  # skip the ~900 MB gene index when nothing is left to filter
            member = self.genes.where_expressed(expressing, mode=mode, min_fraction=min_fraction, validation=None)
            keep &= df["uid"].isin(set(member.to_df()["uid"]))
        params: dict[str, Any] = {
            "validation": validation,
            "license_set": license_set,
            "noncommercial": noncommercial,
            "redistributable": redistributable,
            "search": search,
            "expressing": expressing,
            "mode": mode if expressing is not None else None,
            "min_fraction": min_fraction,
        }
        flt = {k: v for k, v in params.items() if v is not None}
        for k, v in facets.items():
            flt[k] = sorted(v, key=str) if isinstance(v, (list, tuple, set)) else v
        return SpatialDataCohort(df[keep], matched=matched, source=self._source(), filter=flt)

    def search(self, text: str) -> SpatialDataCohort:
        """Deterministic text search over facet values.

        Each token is fuzzy/substring-matched against the distinct values of each facet. A row matches
        a token when any of its facet values matches it, in any facet, so "liver" finds rows whose
        tissue is liver as well as rows whose study_id contains "liver". Tokens are AND-ed:
        "breast xenium" keeps rows that match both. A token that matches no facet value is ignored,
        with one warning that names it; if no token matches anything, the result is an empty
        SpatialDataCohort with a warning. ``matched`` maps each facet to the values that matched any
        token. Rows that failed validation are excluded unless ``validation_status`` itself was matched.
        """
        df = self._df
        tokens = list(dict.fromkeys(re.findall(r"\w+", text.lower())))
        matched: dict[str, list[str]] = {}
        token_masks: dict[str, pd.Series] = {}
        for col in schema.FACETS:
            # DOIs tokenize to "10" etc.; source repeats study_id; booleans hold no searchable text.
            if col in ("doi", "source") or _is_bool_facet(col) or col not in df.columns:
                continue
            values = _distinct(df[col])
            per_token = {t: [v for v in values if _token_matches(t, v)] for t in tokens}
            hit_set = {v for vs in per_token.values() for v in vs}
            if not hit_set:
                continue
            matched[col] = [v for v in values if v in hit_set]
            for t, vs in per_token.items():
                if vs:
                    m = _facet_isin(df[col], vs).fillna(False).astype(bool)
                    token_masks[t] = token_masks[t] | m if t in token_masks else m
        if not matched:
            warnings.warn(f"nothing matched {text!r}", stacklevel=2)
            return SpatialDataCohort(df.iloc[0:0], source=self._source(), filter={"search": text})
        unmatched = [t for t in tokens if t not in token_masks]
        if unmatched:
            warnings.warn(f"search token(s) {unmatched} match no facet value and were ignored", stacklevel=2)
        mask = pd.Series(True, index=df.index)
        for m in token_masks.values():
            mask &= m
        if "validation_status" not in matched:
            mask &= df["validation_status"] == "pass"
        return SpatialDataCohort(
            df[mask.fillna(False).astype(bool)], matched=matched, source=self._source(), filter={"search": text}
        )

    def dataset(self, uid: str) -> Dataset:
        """The catalog row for one dataset, by full ``uid``, its 16-character stem or its ``canonical_key``.

        A uid pins one version. A stem resolves to the latest version. A ``canonical_key`` such as
        ``"geo:GSM9778330"`` is unique where it is set and usually survives a re-version of the dataset; datasets
        without one (null in the column) cannot be looked up this way. Raises ``KeyError`` if absent or ambiguous.
        """
        hit = self._df[self._df["uid"] == uid]
        if hit.empty and len(uid) == 16:
            hit = self._df[self._df["uid"].str[:16] == uid]
            if len(hit) > 1:
                raise KeyError(f"stem {uid!r} matches several uids: {sorted(hit['uid'])}; pass the full uid")
        if hit.empty and "canonical_key" in self._df.columns:
            hit = self._df[self._df["canonical_key"].astype("string") == uid]
            if len(hit) > 1:
                raise KeyError(f"canonical_key {uid!r} matches several uids: {sorted(hit['uid'])}; pass the full uid")
        if hit.empty:
            raise KeyError(f"uid not in catalog: {uid!r}")
        return Dataset(hit.iloc[0], self._source())

    def collection(self, slug: str) -> SpatialDataCohort:
        """The members of one collection as a cohort, by its slug (the last part of its web page URL).

        Members are the catalog rows listing ``slug`` in their ``collections`` column, in catalog order.
        Raises ``KeyError`` if no row lists it.
        """
        if "collections" not in self._df.columns:
            raise ValueError("catalog has no 'collections' column")
        sub = self._df[_facet_isin(self._df["collections"], [slug]).fillna(False).astype(bool)]
        if sub.empty:
            raise KeyError(f"unknown collection {slug!r}: no catalog row lists it")
        return SpatialDataCohort(sub, source=self._source(), filter={"collection": slug})

    def cohort(self, *, uids: list[str] | None = None, study_ids: list[str] | None = None) -> SpatialDataCohort:
        """Build a cohort from an explicit set of ``uids`` or ``study_ids`` (exactly one).

        Raises ``KeyError`` naming any id not in the catalog; a valid but empty result warns.
        """
        if (uids is None) == (study_ids is None):
            raise ValueError("pass exactly one of uids= or study_ids=")
        col, want = ("uid", uids) if uids is not None else ("study_id", study_ids or [])
        if col not in self._df.columns:
            raise ValueError(f"catalog has no {col!r} column")
        want = [str(x) for x in want]
        sub = self._df[self._df[col].astype("string").isin(want)]
        found = set(sub[col].astype("string"))
        missing = [x for x in want if x not in found]
        if missing:
            raise KeyError(f"unknown {col}(s) not in catalog: {missing}")
        if sub.empty:
            warnings.warn(f"no catalog rows for {col} in {want}", stacklevel=2)
        return SpatialDataCohort(sub, source=self._source(), filter={col: sorted(want)})
