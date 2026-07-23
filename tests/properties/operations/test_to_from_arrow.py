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


def arrow_writable(a: ak.Array) -> bool:
    return _nodes_writable(a.layout)


def _nodes_writable(
    layout: ak.contents.Content,
    nullable: bool = False,
    sliced: bool = False,
    lossless: bool = True,
) -> bool:
    """Exclude layouts that `to_arrow` or `from_arrow` currently mishandles.

    `nullable` tracks whether Arrow validity bytes flow into this node from
    an enclosing option: they start at option nodes, pass through records
    and indexed nodes, and stop below lists. A var-length list receiving
    validity bytes whose offsets do not start at zero is compacted against
    unshifted content in `ListOffsetArray._to_arrow`, shifting every list
    by `offsets[0]` (data corruption, #4222).

    `sliced` tracks whether `to_arrow` reaches this node through a
    transformation that can move list offsets away from zero even if they
    were constructed zero-based: a list trims its content to
    `offsets[0]:offsets[length]`, an indexed node projects its content, a
    `ListArray` may be compacted from anywhere, and a union selects each
    child through its index; such nodes can present any list below an
    option with nonzero offsets, triggering #4222.

    `lossless` distinguishes exact reconstruction from mere convertibility:
    layouts whose type the roundtrip changes without crashing or corrupting
    values (mergeable union contents, unknown-type fields under an option)
    are excluded only when `lossless` is true.
    """
    if layout.is_union:
        if nullable:
            # `UnionArray._to_arrow` scatters incoming validity bytes with
            # a per-child index that it misapplies when the index skips
            # values (IndexError, #4228)
            return False
        if lossless and any(
            ak._do.mergeable(x, y, mergebool=False)
            for x, y in itertools.combinations(layout.contents, 2)
        ):
            # `from_arrow` rebuilds unions with `UnionArray.simplified`,
            # which merges mergeable contents (e.g. `union[float64, int64]`
            # reads back as `float64`)
            return False
        return all(_nodes_writable(x, False, True, lossless) for x in layout.contents)
    if layout.is_record:
        return all(
            _nodes_writable(x, nullable, sliced, lossless) for x in layout.contents
        )
    if layout.is_regular:
        # `from_arrow` loses the length of a size-0 fixed-size list (a
        # `3 * 0 * float64` array reads back with length 0, #4229)
        return layout.size > 0 and _nodes_writable(
            layout.content, False, sliced, lossless
        )
    if layout.is_option:
        if isinstance(layout, ak.contents.IndexedOptionArray):
            if layout.length > 0 and layout.content.length == 0:
                # `to_arrow` crashes projecting nulls over a zero-length
                # content (IndexError in `ListOffsetArray._to_arrow`, #4221)
                return False
            return _nodes_writable(layout.content, True, True, lossless)
        return _nodes_writable(layout.content, True, sliced, lossless)
    if layout.is_indexed:
        return _nodes_writable(layout.content, nullable, True, lossless)
    if layout.is_list:
        if nullable and layout.length > 0 and (sliced or layout.starts[0] != 0):
            return False
        if isinstance(layout, ak.contents.ListOffsetArray):
            child_sliced = sliced or bool(layout.offsets[0] != 0)
        else:
            child_sliced = True
        return _nodes_writable(layout.content, False, child_sliced, lossless)
    if layout.is_unknown:
        # Arrow's null type is intrinsically nullable: an unknown-type
        # record field under an option reads back as `?unknown`
        return not (lossless and nullable)
    return True


@suppress_too_slow
@given(
    a=st_ak.constructors.arrays(
        dtypes=arrow_dtypes(),
        allow_indexed_option=False,
        allow_byte_masked=False,
        allow_bit_masked=False,
        allow_unmasked=False,
    ).filter(arrow_writable)
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
    ).filter(arrow_writable)
)
def test_roundtrip_masked(a: ak.Array) -> None:
    """Option-type arrays roundtrip through Arrow validity bitmaps.

    `from_arrow` returns every option layout as a `BitMaskedArray` (or
    `UnmaskedArray`), so content classes are not compared.
    """
    arr = ak.to_arrow(a)
    returned = ak.from_arrow(arr)
    assert ak.array_equal(a, returned, equal_nan=True, same_content_types=False)


def table_writable(a: ak.Array) -> bool:
    layout, is_option = a.layout, False
    while layout.is_option or layout.is_indexed:
        is_option = is_option or layout.is_option
        layout = layout.content
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
    return _nodes_writable(a.layout)


@suppress_too_slow
@given(
    a=st_ak.constructors.arrays(
        dtypes=arrow_dtypes(),
        # a bare unknown type crashes `from_arrow` (AttributeError in
        # `form_remove_optiontype`, #4230), and a nested one breaks the
        # schema comparison
        allow_empty=False,
        allow_union=False,
    ).filter(table_writable)
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


def stable_writable(a: ak.Array) -> bool:
    return _nodes_writable(a.layout, lossless=False)


@suppress_too_slow
@given(
    a=st_ak.constructors.arrays(
        dtypes=arrow_dtypes(),
        # pyarrow's `Array.equals` counts NaN as unequal to itself and has
        # no `equal_nan` option; NaN coverage lives in the tests above
        allow_nan=False,
    ).filter(stable_writable)
)
def test_roundtrip_stable(a: ak.Array) -> None:
    """One roundtrip brings the conversion to a fixed point.

    The roundtrip can lose type information on the write side (an
    unknown-type record field under an option gains an option) or on the
    read side (`from_arrow` merges mergeable union contents), so neither
    the array nor the first Arrow conversion is reproducible in general.
    After one full roundtrip both normalizations have happened: converting
    the reconstruction again yields an identical Arrow array. Layouts that
    `to_arrow` crashes on or corrupts (#4219, #4221, #4222) are still
    excluded: this test does not vouch for them.
    """
    r = ak.from_arrow(ak.to_arrow(a))
    m = ak.to_arrow(r)
    m2 = ak.to_arrow(ak.from_arrow(m))
    assert m2.equals(m)
