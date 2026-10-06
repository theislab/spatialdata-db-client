"""Thin command-line interface over the catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rapidfuzz import fuzz, process

from sddb.catalog import Catalog, facet_values
from sddb.citations import parse_bibtex
from sddb.dataset import Results

app = typer.Typer(help="Query and download spatialdata-db datasets.", no_args_is_help=True, add_completion=False)

_CatalogOpt = Annotated[str | None, typer.Option("--catalog", help="Catalog URL (default: $SDDB_CATALOG_URL).")]
_COLS = ("uid", "technology", "organism", "tissue", "n_obs")


def _fail(err: Exception) -> typer.Exit:
    typer.echo(f"error: {err}", err=True)
    return typer.Exit(1)


def _fmt(v: Any) -> str:
    return "-" if pd.isna(v) else f"{v:.4g}" if isinstance(v, float) else str(v)


def _print_table(cols: list[str], rows: list[list[str]]) -> None:
    widths = [max([len(c), *(len(r[i]) for r in rows)]) for i, c in enumerate(cols)]
    for r in [cols, *rows]:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())


def _facets(**opts: Any) -> dict[str, Any]:
    return {k: v for k, v in opts.items() if v is not None}


def _validation(v: str) -> str | None:
    if v not in ("pass", "all"):
        raise typer.BadParameter("must be 'pass' or 'all'", param_hint="--validation")
    return None if v == "all" else "pass"


@app.command()
def query(
    organism: str | None = None,
    assay: str | None = None,
    tissue: str | None = None,
    disease: str | None = None,
    technology: str | None = None,
    tier: str | None = None,
    validation: Annotated[str, typer.Option(help="'pass' (default) or 'all'.")] = "pass",
    expressing: Annotated[
        str | None, typer.Option(help="Keep datasets expressing this gene symbol (downloads the gene index once).")
    ] = None,
    min_fraction: Annotated[float | None, typer.Option(help="With --expressing: min fraction_obs_detected.")] = None,
    search: Annotated[str | None, typer.Option(help="Free-text search over facet values.")] = None,
    min_obs: Annotated[int | None, typer.Option(help="Keep datasets with n_obs >= this.")] = None,
    min_features: Annotated[int | None, typer.Option(help="Keep datasets with n_features >= this.")] = None,
    catalog: _CatalogOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON records.")] = False,
) -> None:
    """List datasets matching the given facets."""
    val = _validation(validation)
    facets = _facets(organism=organism, assay=assay, tissue=tissue, disease=disease, technology=technology, tier=tier)
    ranges = _facets(n_obs__gte=min_obs, n_features__gte=min_features)
    try:
        cat = Catalog(catalog)
        res = cat.query(
            validation=val, search=search, expressing=expressing, min_fraction=min_fraction, **facets, **ranges
        )
    except Exception as err:
        raise _fail(err) from err

    df = res.to_df()
    if df.empty:
        _suggest(cat, facets)
    if as_json:
        typer.echo(df.to_json(orient="records", indent=2))
        return

    # Print catalog metadata header for table output
    generated_at = cat.generated_at
    date_str = f" {generated_at}" if generated_at else ""
    typer.echo(f"# catalog{date_str} · {len(cat)} datasets", err=True)

    cols = [c for c in _COLS if c in df.columns]
    _print_table(cols, [[_fmt(v) for v in r] for r in df[cols].itertuples(index=False)])


def _norm(v: str) -> str:
    return "".join(str(v).lower().split())


def _suggest(cat: Catalog, facets: dict[str, Any]) -> None:
    """Hint the closest known value for each facet value that matches nothing (exact-match query)."""
    df = cat.to_df()
    for col, val in facets.items():
        if col not in df.columns or isinstance(val, (list, tuple, set)):
            continue
        known = {str(v) for v in df[col].dropna().unique()}
        if str(val) in known:
            continue
        best = process.extractOne(val, list(known), processor=_norm, scorer=fuzz.ratio, score_cutoff=70)
        if best:
            typer.echo(f"no match for --{col} {val!r}; did you mean '{best[0]}'?", err=True)


@app.command()
def facets(
    field: Annotated[str | None, typer.Argument(help="Facet column; omit to list columns.")] = None,
    catalog: _CatalogOpt = None,
) -> None:
    """List facet columns, or the distinct values of FIELD."""
    try:
        cols = facet_values(Catalog(catalog).to_df(), field)
    except Exception as err:
        raise _fail(err) from err
    typer.echo("\n".join(cols))


def _resolve(cat: Catalog, uids: list[str]) -> Results:
    df = cat.to_df()
    missing = [u for u in uids if u not in set(df["uid"])]
    if missing:
        raise ValueError(f"uid(s) not in catalog: {missing}")
    return Results(df[df["uid"].isin(uids)], source=cat._source())


@app.command()
def download(
    uids: Annotated[list[str], typer.Argument(help="Dataset uid(s).")],
    dest: Annotated[Path, typer.Option(help="Destination directory.")] = Path("."),
    catalog: _CatalogOpt = None,
    workers: Annotated[int, typer.Option(help="Parallel downloads.")] = 4,
) -> None:
    """Download datasets to DEST and print the manifest path."""
    try:
        manifest = _resolve(Catalog(catalog), uids).download(dest, workers=workers)
    except Exception as err:
        raise _fail(err) from err
    failed = [e.uid for e in manifest.entries if e.status != "complete"]
    typer.echo(str(dest / "manifest.json"))
    if failed:
        typer.echo(f"error: failed: {failed}", err=True)
        raise typer.Exit(1)


@app.command("viewer-url")
def viewer_url(uid: str, catalog: _CatalogOpt = None) -> None:
    """Print the browser viewer link for a dataset."""
    try:
        typer.echo(_resolve(Catalog(catalog), [uid])[0].viewer_url())
    except Exception as err:
        raise _fail(err) from err


@app.command()
def genes(
    symbol: Annotated[str, typer.Argument(help="Gene symbol (case-insensitive).")],
    limit: Annotated[int, typer.Option(help="Max datasets to list.")] = 20,
    catalog: _CatalogOpt = None,
) -> None:
    """List datasets expressing SYMBOL, ranked by fraction of observations detected.

    The first call downloads the cross-dataset gene index (~105 MB, one-time; cached afterwards).
    """
    try:
        df = Catalog(catalog).genes.ranked(symbol).head(limit)
    except Exception as err:
        raise _fail(RuntimeError(f"could not load the gene index: {err}")) from err
    if df.empty:
        typer.echo(f"no datasets express {symbol!r}", err=True)
        return
    _print_table(list(df.columns), [[_fmt(v) for v in r] for r in df.itertuples(index=False)])


@app.command()
def cite(
    output: Annotated[Path, typer.Option("-o", "--output", help="BibTeX file to write.")],
    organism: str | None = None,
    assay: str | None = None,
    tissue: str | None = None,
    disease: str | None = None,
    technology: str | None = None,
    tier: str | None = None,
    search: Annotated[str | None, typer.Option(help="Free-text search over facet values.")] = None,
    validation: Annotated[str, typer.Option(help="'pass' (default) or 'all'.")] = "pass",
    bib_url: Annotated[str | None, typer.Option(help="citations.bib URL/path (default: next to the catalog).")] = None,
    catalog: _CatalogOpt = None,
) -> None:
    """Write BibTeX for the studies of the datasets matching the given facets/search."""
    val = _validation(validation)
    facets = _facets(organism=organism, assay=assay, tissue=tissue, disease=disease, technology=technology, tier=tier)
    try:
        cat = Catalog(catalog)
        res = cat.query(validation=val, search=search, **facets)
    except Exception as err:
        raise _fail(err) from err
    if len(res) == 0:
        typer.echo("error: no datasets matched", err=True)
        raise typer.Exit(1)
    try:
        res.citations(output, bib_url=bib_url)
    except Exception as err:
        raise _fail(err) from err
    n = len(parse_bibtex(output.read_text(encoding="utf-8")))
    typer.echo(f"wrote {n} citations -> {output}")
