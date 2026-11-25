# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property test that awkward parses every string the type-string grammar generates.

The parser in `ak.types.from_datashape` builds the type while it
parses, so a failure to build the type is a failure to parse. The
parse with `datashape.GRAMMAR`, which only gives the predicates a
tree, cannot fail.
"""

import lark
from hypothesis import assume, given
from hypothesis.extra.lark import from_lark

import awkward as ak
from tests.properties.grammar import datashape


@given(s=from_lark(datashape.GRAMMAR))
def test_properties(s: str) -> None:
    """Assert `ak.types.from_datashape` reads a string the grammar generates."""
    tree = datashape.GRAMMAR.parse(s)

    assume(not _would_raise_from_known_issue(tree))

    ak.types.from_datashape(s, highlevel=False)


def _would_raise_from_known_issue(tree: lark.Tree) -> bool:
    """Return `True` if a known cause makes the string of `tree` fail.

    One clause per known cause.
    """
    if datashape.has_dtype_unavailable_locally(tree):
        return True
    if datashape.has_unknown_with_parameters(tree):
        return True
    if datashape.has_datetime_count_over_int32(tree):
        return True
    if datashape.has_datetime_count_zero(tree):
        return True
    return False
