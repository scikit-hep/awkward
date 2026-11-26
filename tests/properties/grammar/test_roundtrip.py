# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property test that a type read from the type-string grammar prints and reads back.

For a string the grammar generates, `str` of the type that
`ak.types.from_datashape` reads must be in the grammar and must read
as the same type, parameters included.
"""

import lark
from hypothesis import assume, given
from hypothesis.extra.lark import from_lark

import awkward as ak
from tests.properties.grammar import datashape


@given(s=from_lark(datashape.GRAMMAR))
def test_properties(s: str) -> None:
    """Assert the printed type is in the grammar and reads as the same type."""
    tree = datashape.GRAMMAR.parse(s)
    assume(not _would_raise_from_known_issue(tree))
    type_ = ak.types.from_datashape(s, highlevel=False)
    assume(not _would_change_from_known_issue(type_))

    printed = str(type_)
    datashape.GRAMMAR.parse(printed)
    again = ak.types.from_datashape(printed, highlevel=False)

    assert again.is_equal_to(type_, all_parameters=True), (
        f"{printed!r} reads as {again!r}, not {type_!r}"
    )


def _would_raise_from_known_issue(tree: lark.Tree) -> bool:
    """Return `True` if a known cause makes the string of `tree` fail.

    The same causes as in `test_parsable.py`. One clause per known
    cause.
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


def _would_change_from_known_issue(type_: ak.types.Type) -> bool:
    """Return `True` if a known cause makes `type_` fail the round trip.

    One clause per known cause.
    """
    if datashape.has_infinite_parameter(type_):
        return True
    if datashape.has_empty_union_with_parameters(type_):
        return True
    if datashape.has_string_with_parameters(type_):
        return True
    if datashape.has_categorical_option(type_):
        return True
    if datashape.has_duplicate_field_names(type_):
        return True
    return False
