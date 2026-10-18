"""Markdown reference for the ``sddb-cli`` Typer app, via Typer's own docs generator."""

import typer
import typer.cli
import typer.main

from sddb.cli import app


def cli_markdown() -> str:
    """Return the CLI reference as Markdown."""
    cmd = typer.main.get_command(app)
    ctx = typer.Context(cmd, info_name="sddb-cli")
    return typer.cli.get_docs_for_click(obj=cmd, ctx=ctx, name="sddb-cli")
