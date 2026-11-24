# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""The "Defined in … on line N" links of the API reference pages."""

_REPOSITORY_URL = "https://github.com/scikit-hep/awkward"


def source_link(rel_path: str, line: int | None, commit: str) -> str:
    """Return reST that links to the file, and the line if given, on GitHub.

    ``rel_path`` is relative to the repository root, e.g.
    ``src/awkward/operations/ak_flatten.py``; the link shows it as a module
    name, ``awkward.operations.ak_flatten``.
    """
    module_name = rel_path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    base_url = f"{_REPOSITORY_URL}/blob/{commit}/{rel_path}"
    if line:
        return f"Defined in `{module_name} <{base_url}>`__ on `line {line} <{base_url}#L{line}>`__."
    return f"Defined in `{module_name} <{base_url}>`__."


def commit_from_git(returncode: int, stdout: bytes) -> str:
    """Return the commit for the links, from the result of ``git rev-parse HEAD``.

    If git fails or prints nothing, as in a build from a source archive
    without ``.git``, the links point to ``main``.
    """
    commit = stdout.decode("utf-8", errors="replace").strip()
    return commit if returncode == 0 and commit else "main"
