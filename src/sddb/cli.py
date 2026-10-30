"""Thin command-line interface over the catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from sddb.catalog import Catalog, facet_values
from sddb.citations import citation_entries
from sddb.cohort import SpatialDataCohort
from sddb.manifest import reusable

app = typer.Typer(help="Query and download spatialdata-db datasets.", no_args_is_help=True, add_completion=False)

_CatalogOpt = Annotated[str | None, typer.Option("--catalog", help="Catalog URL (default: $SDDB_CATALOG_URL).")]
_COLS = ("uid", "technology", "organism", "tissue", "n_obs")
_SourceOpt = Annotated[str | None, typer.Option(help="Data source: the study_id prefix (geo, hest, 10x, ...).")]
_HneOpt = Annotated[
    bool | None, typer.Option("--hne-image/--no-hne-image", help="Keep datasets with (or without) an H&E image.")
]
_RedistOpt = Annotated[
    bool | None,
    typer.Option(
        "--redistributable/--no-redistributable",
        help="Keep datasets whose known license allows redistribution (or the rest, which includes unknown licenses).",
    ),
]


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
    source: _SourceOpt = None,
    preservation_method: Annotated[
        str | None, typer.Option(help="Preservation method, for example FFPE or fresh frozen.")
    ] = None,
    hne_image: _HneOpt = None,
    redistributable: _RedistOpt = None,
    validation: Annotated[str, typer.Option(help="'pass' (default) or 'all'.")] = "pass",
    gene: Annotated[
        list[str] | None,
        typer.Option(
            "--gene",
            "--expressing",
            help="Gene symbol or Ensembl id; repeat to require all (downloads the gene index once).",
        ),
    ] = None,
    min_fraction: Annotated[float | None, typer.Option(help="With --gene: min fraction_obs_detected.")] = None,
    search: Annotated[str | None, typer.Option(help="Free-text search over facet values.")] = None,
    min_obs: Annotated[int | None, typer.Option(help="Keep datasets with n_obs >= this.")] = None,
    min_features: Annotated[int | None, typer.Option(help="Keep datasets with n_features >= this.")] = None,
    catalog: _CatalogOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON records.")] = False,
) -> None:
    """List datasets matching the given facets.

    A facet value that is not in the catalog resolves through aliases (case, spaces and hyphens are
    ignored; --technology also takes the assay label, e.g. "10x Xenium"), or fails with a "did you mean" hint.
    """
    val = _validation(validation)
    facets = _facets(
        organism=organism,
        assay=assay,
        tissue=tissue,
        disease=disease,
        technology=technology,
        tier=tier,
        source=source,
        preservation_method=preservation_method,
        hne_image=hne_image,
    )
    ranges = _facets(n_obs__gte=min_obs, n_features__gte=min_features)
    try:
        cat = Catalog(catalog)
        res = cat.query(
            validation=val,
            search=search,
            expressing=gene or None,
            mode="all",
            min_fraction=min_fraction,
            redistributable=redistributable,
            **facets,
            **ranges,
        )
    except Exception as err:
        raise _fail(err) from err

    df = res.to_df()
    if as_json:
        typer.echo(df.to_json(orient="records", indent=2))
        return

    # Print catalog metadata header for table output
    generated_at = cat.generated_at
    date_str = f" {generated_at}" if generated_at else ""
    typer.echo(f"# catalog{date_str} · {len(cat)} datasets", err=True)

    cols = [c for c in _COLS if c in df.columns]
    _print_table(cols, [[_fmt(v) for v in r] for r in df[cols].itertuples(index=False)])


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


def _resolve(cat: Catalog, uids: list[str]) -> SpatialDataCohort:
    df = cat.to_df()
    missing = [u for u in uids if u not in set(df["uid"])]
    if missing:
        raise ValueError(f"uid(s) not in catalog: {missing}")
    return SpatialDataCohort(df[df["uid"].isin(uids)], source=cat._source())


@app.command()
def download(
    uids: Annotated[list[str] | None, typer.Argument(help="Dataset uid(s).")] = None,
    collection: Annotated[str | None, typer.Option(help="A collection slug instead of uids.")] = None,
    dest: Annotated[Path, typer.Option(help="Destination directory.")] = Path("."),
    catalog: _CatalogOpt = None,
    workers: Annotated[int, typer.Option(help="Parallel downloads.")] = 4,
) -> None:
    """Download datasets (by uid, or every member of --collection) to DEST and print the manifest path."""
    if bool(uids) == (collection is not None):
        raise _fail(ValueError("pass dataset uid(s) or --collection, not both or neither"))
    try:
        cat = Catalog(catalog)
        cohort = cat.collection(collection) if collection is not None else _resolve(cat, uids or [])
        present = reusable(dest)
        manifest = cohort.download(dest, workers=workers)
    except Exception as err:
        raise _fail(err) from err
    failed = [e.uid for e in manifest.entries if e.status != "complete"]
    done = [e for e in manifest.entries if e.status == "complete"]
    fetched = [e for e in done if present.get(e.uid) != e.zarr_url]
    mb = sum(e.size_bytes or 0 for e in fetched) / 1e6
    summary = f"downloaded {len(fetched)} dataset{'' if len(fetched) == 1 else 's'} ({mb:,.1f} MB) to {dest}"
    if len(done) > len(fetched):
        summary += f"; {len(done) - len(fetched)} already present"
    typer.echo(str(dest / "manifest.json"))
    typer.echo(summary, err=True)
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
    genes: Annotated[list[str], typer.Argument(help="Gene symbol(s) or Ensembl id(s), case-insensitive.")],
    limit: Annotated[int, typer.Option(help="Max datasets to list.")] = 20,
    validation: Annotated[str, typer.Option(help="'all' (default, includes bronze) or 'pass'.")] = "all",
    catalog: _CatalogOpt = None,
) -> None:
    """List datasets expressing all GENES (one gene: ranked by fraction of observations detected).

    The first call downloads the cross-dataset gene index (~900 MB, one-time; cached and read in batches).
    """
    val = _validation(validation)
    try:
        cat = Catalog(catalog)
        flags = [c for c in ("validation_status", "tier") if c in cat.to_df().columns]
        if len(genes) == 1:
            df = cat.genes.ranked(genes[0], validation=val).head(limit)
            df = df.merge(cat.to_df()[["uid", *flags]], on="uid", how="left")
        else:
            df = cat.genes.where_expressed(genes, mode="all", validation=val).to_df().head(limit)
            df = df[[c for c in (*_COLS, *flags) if c in df.columns]]
    except Exception as err:
        raise _fail(RuntimeError(f"could not load the gene index: {err}")) from err
    if df.empty:
        typer.echo(f"no datasets express {' + '.join(genes)!r}", err=True)
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
    source: _SourceOpt = None,
    hne_image: _HneOpt = None,
    redistributable: _RedistOpt = None,
    search: Annotated[str | None, typer.Option(help="Free-text search over facet values.")] = None,
    validation: Annotated[str, typer.Option(help="'pass' (default) or 'all'.")] = "pass",
    bib_url: Annotated[str | None, typer.Option(help="citations.bib URL/path (default: next to the catalog).")] = None,
    catalog: _CatalogOpt = None,
) -> None:
    """Write BibTeX for the studies of the datasets matching the given facets/search."""
    val = _validation(validation)
    facets = _facets(
        organism=organism,
        assay=assay,
        tissue=tissue,
        disease=disease,
        technology=technology,
        tier=tier,
        source=source,
        hne_image=hne_image,
    )
    try:
        cat = Catalog(catalog)
        res = cat.query(validation=val, search=search, redistributable=redistributable, **facets)
    except Exception as err:
        raise _fail(err) from err
    if len(res) == 0:
        typer.echo("error: no datasets matched", err=True)
        raise typer.Exit(1)
    try:
        text, n_published, n_fallback = citation_entries(res, bib_url=bib_url)
    except Exception as err:
        raise _fail(err) from err
    output.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {n_published + n_fallback} citations ({n_fallback} fallback) -> {output}")
