"""Resolve adapter *roles* (regexes) to a store's element paths."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


def resolve(names: Iterable[str], roles: Mapping[str, str]) -> dict[str, str]:
    """Resolve each role to exactly one element path.

    Parameters
    ----------
    names
        Element paths as ``"kind/name"`` (the keys of :meth:`sddb.Dataset.elements`).
    roles
        Role name to regex, matched with :func:`re.search` against each path.

    Returns
    -------
    dict
        Role name to the single matching path.

    Raises
    ------
    KeyError
        If a role matches nothing; the message lists the paths of the kind the pattern targets
        (a leading ``^kind/``), or all paths when the pattern names no kind.
    ValueError
        If a role matches more than one path; the message lists them.
    """
    paths = list(names)
    out: dict[str, str] = {}
    for role, pattern in roles.items():
        hits = [p for p in paths if re.search(pattern, p)]
        if not hits:
            kind = re.match(r"\^(\w+)/", pattern)
            candidates = [p for p in paths if kind is None or p.startswith(kind.group(1) + "/")]
            raise KeyError(f"role {role!r} (/{pattern}/) matches no element; candidates: {candidates}")
        if len(hits) > 1:
            raise ValueError(f"role {role!r} (/{pattern}/) is ambiguous; matches: {hits}")
        out[role] = hits[0]
    return out
