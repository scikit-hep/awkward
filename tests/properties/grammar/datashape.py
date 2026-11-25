# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""The type-string grammar and predicates for its test modules.

The grammar is `studies/type-parser-2/awkward_datashape_grammar.g`,
the source of `awkward/types/_awkward_datashape_parser.py`.

A predicate for a cause in reading a string takes the parse tree from
`GRAMMAR`. A predicate for a cause in printing or comparing a type
takes the type that `ak.types.from_datashape` returns.
"""

import math
import re
from collections.abc import Iterator
from pathlib import Path

import lark

import awkward as ak
from awkward.types.numpytype import is_primitive

GRAMMAR_PATH = (
    Path(__file__).parents[3] / "studies/type-parser-2/awkward_datashape_grammar.g"
)

# Without placeholders, an empty `json_object` has no children.
GRAMMAR = lark.Lark(
    GRAMMAR_PATH.read_text(encoding="utf-8"),
    parser="lalr",
    maybe_placeholders=False,
)


def has_dtype_unavailable_locally(tree: lark.Tree) -> bool:
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


def has_unknown_with_parameters(tree: lark.Tree) -> bool:
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


def has_datetime_count_over_int32(tree: lark.Tree) -> bool:
    """Return `True` if a `datetime64` or `timedelta64` count is above 2**31 - 1.

    The `DATETIME64` and `TIMEDELTA64` terminals accept a count of any
    size, but NumPy stores it as a 32-bit signed integer. A larger
    count raises `TypeError: Invalid datetime metadata string` from
    NumPy (#4526). A field name is not affected.
    """
    return any(count > _MAX_DATETIME_COUNT for count in _datetime_counts(tree))


def has_datetime_count_zero(tree: lark.Tree) -> bool:
    """Return `True` if a `datetime64` or `timedelta64` count is 0.

    The `DATETIME64` and `TIMEDELTA64` terminals accept the count 0,
    as in `datetime64[0s]`, and `from_datashape` reads it. On x86-64,
    NumPy stops the process with `SIGFPE` when it compares such a
    dtype with one of the same unit and no count, such as
    `datetime64[s]` (#4569). `from_datashape` makes this comparison
    when the other dtype is in the same string or was read earlier in
    the process, so the predicate skips every count of 0. A field name
    is not affected.
    """
    return any(count == 0 for count in _datetime_counts(tree))


def _datetime_counts(tree: lark.Tree) -> Iterator[int]:
    """Yield the count of each `datetime64` or `timedelta64` dtype in `tree`."""
    for name in tree.find_data("numpytype_name"):
        (token,) = name.children
        if token.type in ("DATETIME64", "TIMEDELTA64"):
            match = _DATETIME_COUNT.search(token)
            if match is not None:
                yield int(match.group(1))


def has_infinite_parameter(type_: ak.types.Type) -> bool:
    """Return `True` if a parameter value has an infinite number.

    `SIGNED_NUMBER` accepts a number beyond the range of a float, such
    as `1e400`, and `from_datashape` reads it as `inf`. `str` prints it
    as `Infinity`, which is not JSON and not in the grammar, so reading
    the printed string raises `UnexpectedToken` (#4567).
    """
    return any(
        _has_infinity(value)
        for node in _walk(type_)
        for value in node.parameters.values()
    )


def _has_infinity(value: object) -> bool:
    """Return `True` if `value`, a JSON value, has an infinite number in it."""
    if isinstance(value, float):
        return math.isinf(value)
    if isinstance(value, list):
        return any(_has_infinity(x) for x in value)
    if isinstance(value, dict):
        return any(_has_infinity(x) for x in value.values())
    return False


def has_empty_union_with_parameters(type_: ak.types.Type) -> bool:
    """Return `True` if a union with no contents has parameters.

    `str` prints such a union as `union[, parameters={...}]`, which is
    not in the grammar, so reading the printed string raises
    `UnexpectedToken` (#4568).
    """
    return any(
        isinstance(node, ak.types.UnionType)
        and len(node.contents) == 0
        and node._str_parameters() is not None
        for node in _walk(type_)
    )


_STRING_ARRAYS = ("string", "bytestring", "char", "byte")


def has_string_with_parameters(type_: ak.types.Type) -> bool:
    """Return `True` if a string type has other parameters or contents.

    `str` prints a type whose `__array__` parameter is `"string"`,
    `"bytestring"`, `"char"`, or `"byte"` as the keyword for it, such
    as `string`, and drops its other parameters (#4566). For `string`
    and `bytes`, it also drops the content. Reading the printed string
    gives a type without them. Any type with these values counts,
    unless it is the type that the keyword reads as.
    """
    return any(
        node.parameter("__array__") in _STRING_ARRAYS and not _is_keyword_type(node)
        for node in _walk(type_)
    )


def _is_keyword_type(node: ak.types.Type) -> bool:
    """Return `True` if `node` is the type that a string keyword reads as."""
    array = node.parameter("__array__")
    if array in ("char", "byte"):
        return (
            isinstance(node, ak.types.NumpyType)
            and node.primitive == "uint8"
            and node.parameters == {"__array__": array}
        )
    if isinstance(node, (ak.types.ListType, ak.types.RegularType)):
        content = {"string": "char", "bytestring": "byte"}[array]
        return (
            node.parameters == {"__array__": array}
            and _is_keyword_type(node.content)
            and node.content.parameter("__array__") == content
        )
    return False


def has_categorical_option(type_: ak.types.Type) -> bool:
    """Return `True` if an option type is categorical.

    `categorical[type=?bool]` sets `__categorical__` on the option
    type, and `str` prints it as `?categorical[type=bool]`, which reads
    as a categorical `bool` inside an option type (#4517).
    """
    return any(
        isinstance(node, ak.types.OptionType)
        and node.parameter("__categorical__") is not None
        for node in _walk(type_)
    )


def has_duplicate_field_names(type_: ak.types.Type) -> bool:
    """Return `True` if a record has two fields with the same name.

    Such a record is not equal to itself after the round trip, as in
    `{x: int64, x: bool}` (#4518).
    """
    return any(
        isinstance(node, ak.types.RecordType)
        and not node.is_tuple
        and len(set(node.fields)) != len(node.fields)
        for node in _walk(type_)
    )


def _walk(type_: ak.types.Type) -> Iterator[ak.types.Type]:
    """Yield `type_` and every type in it."""
    yield type_
    if isinstance(type_, (ak.types.RecordType, ak.types.UnionType)):
        for content in type_.contents:
            yield from _walk(content)
    elif hasattr(type_, "content"):
        yield from _walk(type_.content)
