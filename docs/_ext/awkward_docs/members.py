# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Which objects sphinx-autoapi documents."""

import re


def skip_member(what: str, name: str, skip: bool) -> bool:
    """Return whether autoapi skips an object.

    ``what`` is autoapi's type of the object (``"module"``, ``"function"``,
    ...), ``name`` its full name, and ``skip`` autoapi's own decision, which
    applies unless one of these rules does:

    - the top-level ``awkward`` package is documented
    - module-level dunder functions, e.g. ``awkward.__dir__``, are skipped
    - modules and packages with a private component, e.g. ``awkward._do``,
      are skipped; private methods and attributes of public classes are not
    - lowercase submodules of ``awkward.contents``, ``awkward.types``, and
      ``awkward.forms`` are skipped; their classes are documented at the
      package level, and a page for the module would collide with the class
      page on a case-insensitive file system
    """
    if name == "awkward":
        return False
    if what == "function":
        short = name.rsplit(".", 1)[-1]
        if short.startswith("__") and short.endswith("__"):
            return True
    if what in ("module", "package"):
        if any(part.startswith("_") for part in name.split(".")):
            return True
    if what == "module":
        child = name.rsplit(".", 1)[-1]
        if re.match(r"awkward\.(contents|types|forms)\.\w+$", name) and child.islower():
            return True
    return skip
