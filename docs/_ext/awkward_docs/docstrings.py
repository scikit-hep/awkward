# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Conversion of the markup in awkward's docstrings to reST."""

import re
from collections.abc import Collection

# Only URLs and relative .html paths are link targets; this leaves code such
# as ``class GenProto[T](Protocol):`` unchanged
_MARKDOWN_LINK = re.compile(
    r"\[([^\]]*)\]\(((?:https?://|\.\.?/)[^\)]*|[^\)\s]*\.html(?:#[^\)]*)?)\)"
)


def process_docstring(
    docstring: str, member_names: Collection[str], qualname: str
) -> str:
    """Convert the markup in a docstring to reST.

    Line by line:

    - a single backtick becomes a double backtick (an inline literal); a
      double backtick stays as it is
    - ``<<<name>>>`` becomes a reference to the hyperlink target ``name``
    - ``#ak.name`` becomes a cross-reference to ``ak.name``
    - ``#name``, where ``name`` is one of ``member_names``, becomes a
      cross-reference to ``qualname.name``
    - a Markdown link whose target is a URL or an ``.html`` path becomes a
      reST link
    """
    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        line = re.sub(r"(?<!`)`(?!`)", "``", line)
        line = re.sub(r"<<<([^>]+)>>>", r"`\1`_", line)
        line = re.sub(r"#(ak\.[A-Za-z0-9_\.]*[A-Za-z0-9_])", r":py:obj:`\1`", line)
        if member_names:

            def _replace_member(m, _members=member_names, _qn=qualname):
                member = m.group(1)
                if member in _members:
                    return f":py:meth:`{member} <{_qn}.{member}>`"
                return m.group(0)

            line = re.sub(r"#([A-Za-z_][A-Za-z0-9_]*)", _replace_member, line)
        line = _MARKDOWN_LINK.sub(r"`\1 <\2>`__", line)
        lines[i] = line
    return "\n".join(lines)
