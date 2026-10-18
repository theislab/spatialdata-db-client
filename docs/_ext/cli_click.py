"""Expose the Typer app to ``.. click::`` (sphinx-click).

Typer >= 0.27 vendors its own click, so ``typer.main.get_command(app)`` is no longer an
instance of the ``click`` package that sphinx-click checks for. This mirrors the command
tree into plain ``click`` objects; it is documentation-only and never executed.
"""

import click
import typer.main

from sddb.cli import app

_TYPES = {
    "IntParamType": click.INT,
    "FloatParamType": click.FLOAT,
    "BoolParamType": click.BOOL,
    "Path": click.Path(),
}


def _param(p):
    kind = _TYPES.get(type(p.type).__name__, click.STRING)
    kw = {"type": kind, "required": p.required, "default": p.default, "help": getattr(p, "help", None),
          "metavar": p.metavar, "envvar": p.envvar}
    if hasattr(p, "is_flag"):  # option
        decls = [*p.opts, *p.secondary_opts]
        if p.secondary_opts:
            decls = [f"{p.opts[0]}/{p.secondary_opts[0]}"]
        return click.Option(decls, is_flag=p.is_flag, multiple=p.multiple, show_default=bool(getattr(p, "show_default", False)), **kw)
    kw.pop("help")
    return click.Argument([p.name], nargs=-1 if p.multiple else 1, **kw)


def _convert(cmd):
    params = [_param(p) for p in cmd.params]
    if hasattr(cmd, "commands"):
        group = click.Group(cmd.name, help=cmd.help, params=params, hidden=cmd.hidden)
        for sub in cmd.commands.values():
            group.add_command(_convert(sub))
        return group
    return click.Command(cmd.name, help=cmd.help, params=params, hidden=cmd.hidden)


cli = _convert(typer.main.get_command(app))
