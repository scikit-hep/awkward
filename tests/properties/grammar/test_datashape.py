# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property tests for the type-string grammar.

The grammar is `studies/type-parser-2/awkward_datashape_grammar.g`,
the source of `awkward/types/_awkward_datashape_parser.py`. Strings
are generated from it and read with `ak.types.from_datashape`.
"""

import re
from pathlib import Path

import lark
from hypothesis import assume, given
from hypothesis.extra.lark import from_lark

import awkward as ak
from awkward.types.numpytype import is_primitive

_GRAMMAR_PATH = (
    Path(__file__).parents[3] / "studies/type-parser-2/awkward_datashape_grammar.g"
)

_GRAMMAR = lark.Lark(_GRAMMAR_PATH.read_text(encoding="utf-8"), parser="lalr")


@given(s=from_lark(_GRAMMAR))
def test_properties(s: str) -> None:
    """Assert `ak.types.from_datashape` reads a string the grammar generates."""
    tree = _GRAMMAR.parse(s)

    assume(not _would_raise_from_known_issue(tree))

    ak.types.from_datashape(s, highlevel=False)


def _would_raise_from_known_issue(tree: lark.Tree) -> bool:
    """Return `True` if a known cause makes the string of `tree` fail.

    One clause per known cause.
    """
    if _has_dtype_unavailable_locally(tree):
        return True
    if _has_unknown_with_parameters(tree):
        return True
    if _has_datetime_count_over_int32(tree):
        return True
    return False


def _has_dtype_unavailable_locally(tree: lark.Tree) -> bool:
    """Return `True` if a dtype the local NumPy lacks is in the type.

    The `DTYPE` terminal lists every dtype that `NumpyType` knows on
    any platform, and `NumpyType` raises `TypeError: unrecognized
    primitive` for those the local NumPy lacks, such as `float128` on
    macOS (#4390). A grammar that does not depend on the platform
    cannot exclude them. A dtype name as a field name is not affected.
    """
    for name in tree.find_data("numpytype_name"):
        (token,) = name.children
        if token.type == "DTYPE" and not is_primitive(token):
            return True
    return False


def _has_unknown_with_parameters(tree: lark.Tree) -> bool:
    """Return `True` if `list_parameters` gives parameters to an `UnknownType`.

    The `list_parameters` rule, `[type, parameters={...}]`, accepts any
    type, and `UnknownType` raises `TypeError: UnknownType cannot
    contain parameters` if the merged parameters are not empty (#4527).
    `categorical[type=unknown]` sets a parameter on the `UnknownType`,
    so `[categorical[type=unknown], parameters={}]` raises too.
    """
    for node in tree.find_data("list_parameters"):
        content, parameters = node.children
        has_parameters = _unknown_has_parameters(content)
        if has_parameters is not None and (has_parameters or parameters.children):
            return True
    return False


def _unknown_has_parameters(type_tree: lark.Tree) -> bool | None:
    """Return whether the `UnknownType` read from `type_tree` has parameters.

    `None` if the type read from `type_tree` is not an `UnknownType`.
    """
    (node,) = type_tree.children
    if node.data == "unknowntype":
        return False
    if node.data == "categorical":
        (content,) = node.children
        return None if _unknown_has_parameters(content) is None else True
    if node.data == "list_parameters":
        content, parameters = node.children
        has_parameters = _unknown_has_parameters(content)
        if has_parameters is None:
            return None
        return has_parameters or bool(parameters.children)
    return None


_MAX_DATETIME_COUNT = 2**31 - 1

_DATETIME_COUNT = re.compile(r"\[\s*([0-9]+)")


def _has_datetime_count_over_int32(tree: lark.Tree) -> bool:
    """Return `True` if a `datetime64` or `timedelta64` count is above 2**31 - 1.

    The `DATETIME64` and `TIMEDELTA64` terminals accept a count of any
    size, but NumPy stores it as a 32-bit signed integer. A larger
    count raises `TypeError: Invalid datetime metadata string` from
    NumPy (#4526). A field name is not affected.
    """
    for name in tree.find_data("numpytype_name"):
        (token,) = name.children
        if token.type in ("DATETIME64", "TIMEDELTA64"):
            match = _DATETIME_COUNT.search(token)
            if match is not None and int(match.group(1)) > _MAX_DATETIME_COUNT:
                return True
    return False
