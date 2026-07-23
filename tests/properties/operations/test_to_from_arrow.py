# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools

import hypothesis_awkward.strategies as st_ak
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis_awkward.util.dtype import SUPPORTED_DTYPES

import awkward as ak

pytest.importorskip("pyarrow")

# An unlucky draw sequence can spend ~1 s in discarded generation attempts
# (entropy-overrun retries inside hypothesis-awkward's recursive strategies),
# tripping the health check. Hypothesis' built-in "ci" profile, from which
# this suite's profiles inherit on GitHub Actions, already suppresses it.
suppress_too_slow = settings(suppress_health_check=[HealthCheck.too_slow])


def arrow_dtype(dtype: np.dtype) -> bool:
    if dtype.kind == "c":
        # pyarrow has no complex numbers (ArrowNotImplementedError)
        return False
    if dtype.kind == "f" and dtype.itemsize == 16:
        # pyarrow has no extended-precision floats, which exist on some
        # platforms (e.g. float128 on Linux)
        return False
    if dtype.kind == "M":
        # units coarser than "s" are unsupported, except "D", whose values
        # `to_arrow` corrupts (int64 buffer reinterpreted as date32, #4219)
        return np.datetime_data(dtype)[0] in {"s", "ms", "us", "ns"}
    if dtype.kind == "m":
        # pyarrow durations support only seconds and finer units
        return np.datetime_data(dtype)[0] in {"s", "ms", "us", "ns"}
    return True


SUPPORTED_ARROW_DTYPES = tuple(d for d in SUPPORTED_DTYPES if arrow_dtype(d))


def arrow_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for dtypes that survive the Arrow roundtrip."""
    return st.sampled_from(SUPPORTED_ARROW_DTYPES)


def arrow_compatible(a: ak.Array) -> bool:
    """Arrays whose type the Arrow conversion preserves.

    The clauses in `_loses_type` are limits of the conversion itself and
    stay even when every cited issue is fixed.
    """
    return not _loses_type(a.layout)


def _loses_type(layout: ak.contents.Content, nullable: bool = False) -> bool:
    """The roundtrip changes the array's type without corrupting values.

    `nullable` tracks whether Arrow validity bytes flow into this node
    from an enclosing option: they start at option nodes, pass through
    records and indexed nodes, and stop below lists.
    """
    if layout.is_union:
        if any(
            ak._do.mergeable(x, y, mergebool=False)
            for x, y in itertools.combinations(layout.contents, 2)
        ):
            # `from_arrow` rebuilds unions with `UnionArray.simplified`,
            # which merges mergeable contents (e.g. `union[float64, int64]`
            # reads back as `float64`)
            return True
        return any(_loses_type(x, False) for x in layout.contents)
    if layout.is_record:
        return any(_loses_type(x, nullable) for x in layout.contents)
    if layout.is_regular:
        return _loses_type(layout.content, False)
    if layout.is_option:
        return _loses_type(layout.content, True)
    if layout.is_indexed:
        return _loses_type(layout.content, nullable)
    if layout.is_list:
        return _loses_type(layout.content, False)
    if layout.is_unknown:
        # Arrow's null type is intrinsically nullable: an unknown-type
        # record field under an option reads back as `?unknown`
        return nullable
    return False


def has_issues(a: ak.Array) -> bool:
    """Arrays affected by a known issue in the conversion.

    One function per known issue, each named after the issue whose
    released fix deletes it; with all of them gone the test filters
    reduce to `arrow_compatible` and `table_compatible`.
    """
    if _has_issue_4221(a.layout):
        return True
    if _has_issue_4222(a.layout):
        return True
    if _has_issue_4228(a.layout):
        return True
    if _has_issue_4229(a.layout):
        return True
    return False


def _has_issue_4221(layout: ak.contents.Content) -> bool:
    """`to_arrow` crashes projecting nulls over a zero-length content
    (IndexError in `ListOffsetArray._to_arrow`, #4221)."""
    if (
        isinstance(layout, ak.contents.IndexedOptionArray)
        and layout.length > 0
        and layout.content.length == 0
    ):
        return True
    return any(_has_issue_4221(x) for x in _children(layout))


def _has_issue_4222(
    layout: ak.contents.Content,
    nullable: bool = False,
    sliced: bool = False,
) -> bool:
    """A nullable var-length list whose offsets do not start at zero is
    compacted against unshifted content in `ListOffsetArray._to_arrow`,
    shifting every list by `offsets[0]` (data corruption, #4222).

    `nullable` tracks whether Arrow validity bytes flow into this node
    from an enclosing option: they start at option nodes, pass through
    records and indexed nodes, and stop below lists.

    `sliced` tracks whether `to_arrow` reaches this node through a
    transformation that can move list offsets away from zero even if
    they were constructed zero-based: a list trims its content to
    `offsets[0]:offsets[length]`, an indexed node projects its content,
    a `ListArray` may be compacted from anywhere, and a union selects
    each child through its index; such nodes can present any list below
    an option with nonzero offsets.
    """
    if layout.is_union:
        return any(_has_issue_4222(x, False, True) for x in layout.contents)
    if layout.is_record:
        return any(_has_issue_4222(x, nullable, sliced) for x in layout.contents)
    if layout.is_regular:
        return _has_issue_4222(layout.content, False, sliced)
    if layout.is_option:
        if isinstance(layout, ak.contents.IndexedOptionArray):
            return _has_issue_4222(layout.content, True, True)
        return _has_issue_4222(layout.content, True, sliced)
    if layout.is_indexed:
        return _has_issue_4222(layout.content, nullable, True)
    if layout.is_list:
        if nullable and layout.length > 0 and (sliced or layout.starts[0] != 0):
            return True
        if isinstance(layout, ak.contents.ListOffsetArray):
            child_sliced = sliced or bool(layout.offsets[0] != 0)
        else:
            child_sliced = True
        return _has_issue_4222(layout.content, False, child_sliced)
    return False


def _has_issue_4228(layout: ak.contents.Content, nullable: bool = False) -> bool:
    """`UnionArray._to_arrow` scatters incoming validity bytes with a
    per-child index that it misapplies when the index skips values
    (IndexError, #4228).

    `nullable` tracks validity bytes as in `_has_issue_4222`.
    """
    if layout.is_union:
        if nullable:
            return True
        return any(_has_issue_4228(x, False) for x in layout.contents)
    if layout.is_record:
        return any(_has_issue_4228(x, nullable) for x in layout.contents)
    if layout.is_regular:
        return _has_issue_4228(layout.content, False)
    if layout.is_option:
        return _has_issue_4228(layout.content, True)
    if layout.is_indexed:
        return _has_issue_4228(layout.content, nullable)
    if layout.is_list:
        return _has_issue_4228(layout.content, False)
    return False


def _has_issue_4229(layout: ak.contents.Content) -> bool:
    """`from_arrow` loses the length of a size-0 fixed-size list (a
    `3 * 0 * float64` array reads back with length 0, #4229)."""
    if layout.is_regular and layout.size == 0:
        return True
    return any(_has_issue_4229(x) for x in _children(layout))


def _children(layout: ak.contents.Content) -> list[ak.contents.Content]:
    """The direct child layouts of a node."""
    if layout.is_record or layout.is_union:
        return list(layout.contents)
    if layout.is_option or layout.is_indexed or layout.is_list:
        return [layout.content]
    return []


def _strip_options(
    layout: ak.contents.Content,
) -> tuple[ak.contents.Content, bool]:
    """Strip option and indexed wrappers, reporting whether an option
    was among them."""
    is_option = False
    while layout.is_option or layout.is_indexed:
        is_option = is_option or layout.is_option
        layout = layout.content
    return layout, is_option


@suppress_too_slow
@given(
    a=st_ak.constructors.arrays(
        dtypes=arrow_dtypes(),
        allow_indexed_option=False,
        allow_byte_masked=False,
        allow_bit_masked=False,
        allow_unmasked=False,
    ).filter(lambda a: arrow_compatible(a) and not has_issues(a))
)
def test_roundtrip(a: ak.Array) -> None:
    """`to_arrow` followed by `from_arrow` reconstructs the array."""
    arr = ak.to_arrow(a)
    returned = ak.from_arrow(arr)
    assert ak.array_equal(a, returned, equal_nan=True)


@suppress_too_slow
@given(
    a=st_ak.constructors.arrays(
        dtypes=arrow_dtypes(),
        # a union with an option-type child that is longer than its last
        # index reference crashes `to_arrow` (IndexError in
        # `ListOffsetArray._to_arrow` via `UnionArray._to_arrow`, #4228)
        allow_union=False,
    ).filter(lambda a: arrow_compatible(a) and not has_issues(a))
)
def test_roundtrip_masked(a: ak.Array) -> None:
    """Option-type arrays roundtrip through Arrow validity bitmaps.

    `from_arrow` returns every option layout as a `BitMaskedArray` (or
    `UnmaskedArray`), so content classes are not compared.
    """
    arr = ak.to_arrow(a)
    returned = ak.from_arrow(arr)
    assert ak.array_equal(a, returned, equal_nan=True, same_content_types=False)


def table_compatible(a: ak.Array) -> bool:
    """Arrays that additionally survive the table wrapping of
    `to_arrow_table`.

    Like `arrow_compatible`, which this extends, the clauses here are
    limits of the representation and stay even when every cited issue is
    fixed.
    """
    layout, is_option = _strip_options(a.layout)
    if layout.is_record:
        if layout.fields == [""]:
            # collides with the anonymous column that `to_arrow_table` uses
            # for non-record arrays; reads back unwrapped
            return False
        if not layout.is_tuple and len(layout.fields) == 0:
            # becomes a zero-column table, losing the length
            return False
        if is_option and all(x.is_option for x in layout.contents):
            # a valid row whose fields are all null reads back as a null
            # row (outermost struct validity is not stored)
            return False
    return arrow_compatible(a)


@suppress_too_slow
@given(
    a=st_ak.constructors.arrays(
        dtypes=arrow_dtypes(),
        # a bare unknown type crashes `from_arrow` (AttributeError in
        # `form_remove_optiontype`, #4230), and a nested one breaks the
        # schema comparison
        allow_empty=False,
        allow_union=False,
    ).filter(lambda a: table_compatible(a) and not has_issues(a))
)
def test_roundtrip_table(a: ak.Array) -> None:
    """Arrays roundtrip through `to_arrow_table`, and `from_arrow_schema`
    predicts the form that `from_arrow` returns.

    Without `generate_bitmasks=True`, an all-valid nullable field reads
    back as an `UnmaskedArray`, whereas the schema, which cannot know
    whether a validity buffer is present, converts to a `BitMaskedArray`
    form.
    """
    table = ak.to_arrow_table(a)
    returned = ak.from_arrow(table, generate_bitmasks=True)
    assert ak.array_equal(a, returned, equal_nan=True, same_content_types=False)
    assert ak.from_arrow_schema(table.schema) == returned.layout.form


@suppress_too_slow
@given(
    a=st_ak.constructors.arrays(
        dtypes=arrow_dtypes(),
        # pyarrow's `Array.equals` counts NaN as unequal to itself and has
        # no `equal_nan` option; NaN coverage lives in the tests above
        allow_nan=False,
    ).filter(lambda a: not has_issues(a))
)
def test_roundtrip_stable(a: ak.Array) -> None:
    """One roundtrip brings the conversion to a fixed point.

    The roundtrip can lose type information on the write side (an
    unknown-type record field under an option gains an option) or on the
    read side (`from_arrow` merges mergeable union contents), so neither
    the array nor the first Arrow conversion is reproducible in general.
    After one full roundtrip both normalizations have happened: converting
    the reconstruction again yields an identical Arrow array. Layouts
    affected by `has_issues` (or by #4219, in the dtype filter) are still
    excluded: this test does not vouch for them.
    """
    r = ak.from_arrow(ak.to_arrow(a))
    m = ak.to_arrow(r)
    m2 = ak.to_arrow(ak.from_arrow(m))
    assert m2.equals(m)
