"""Thin command-line interface over the catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from sddb.catalog import Catalog
from sddb.dataset import Results

app = typer.Typer(help="Query and download spatialdata-db datasets.", no_args_is_help=True, add_completion=False)

_CatalogOpt = Annotated[str | None, typer.Option("--catalog", help="Catalog URL (default: $SDDB_CATALOG_URL).")]
_COLS = ("uid", "technology", "organism", "tissue", "n_obs")


def _fail(err: Exception) -> typer.Exit:
    typer.echo(f"error: {err}", err=True)
    return typer.Exit(1)


@app.command()
def query(
    organism: str | None = None,
    assay: str | None = None,
    tissue: str | None = None,
    disease: str | None = None,
    technology: str | None = None,
    tier: str | None = None,
    validation: Annotated[str, typer.Option(help="'pass' (default) or 'all'.")] = "pass",
    catalog: _CatalogOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON records.")] = False,
) -> None:
    """List datasets matching the given facets."""
    if validation not in ("pass", "all"):
        raise typer.BadParameter("must be 'pass' or 'all'", param_hint="--validation")
    given = {"organism": organism, "assay": assay, "tissue": tissue, "disease": disease}
    given |= {"technology": technology, "tier": tier}
    facets: dict[str, Any] = {k: v for k, v in given.items() if v is not None}
    try:
        cat = Catalog(catalog)
        res = cat.query(validation=None if validation == "all" else "pass", **facets)
    except Exception as err:
        raise _fail(err) from err

    df = res.to_df()
    if as_json:
        typer.echo(df.to_json(orient="records", indent=2))
        return

    # Print catalog metadata header for table output
    date_str = f" {cat.generated_at}" if cat.generated_at else ""
    typer.echo(f"# catalog{date_str} · {len(cat)} datasets", err=True)

    cols = [c for c in _COLS if c in df.columns]
    rows = [["-" if pd.isna(v) else str(v) for v in r] for r in df[cols].itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    for r in [cols, *rows]:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())


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
