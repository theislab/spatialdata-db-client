"""Remove absolute paths from an executed notebook's metadata, in place.

    python scripts/clean_notebook.py <bundle>/primer.ipynb

papermill records the input and output paths of a run in the notebook's metadata; on the HPC those name
the workspace. Every metadata string that is an absolute path is removed, in the notebook's and in each
cell's metadata. Cell sources and outputs are left as they are.
"""

from __future__ import annotations

import re
import sys

import nbformat

_ABSOLUTE = re.compile(r"^(/|[A-Za-z]:[\\/]|~/)")


def strip_paths(value: object) -> object:
    """``value`` with every dict entry whose value is an absolute path string removed, recursively."""
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items() if not (isinstance(v, str) and _ABSOLUTE.match(v))}
    if isinstance(value, list):
        return [strip_paths(v) for v in value if not (isinstance(v, str) and _ABSOLUTE.match(v))]
    return value


def main(argv: list[str]) -> int:
    """Clean each notebook named in ``argv``."""
    for path in argv:
        nb = nbformat.read(path, as_version=4)
        nb.metadata = nbformat.from_dict(strip_paths(nb.metadata))
        for cell in nb.cells:
            cell.metadata = nbformat.from_dict(strip_paths(cell.metadata))
        nbformat.write(nb, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
