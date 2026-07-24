# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import threading
from pathlib import Path

import hypothesis_awkward.strategies as st_ak
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis_awkward.util.dtype import SUPPORTED_DTYPES

import awkward as ak

pytest.importorskip("pyarrow.feather")

# The first example pays one-time lazy initialization inside the first
# `to_feather` call (~320 ms: pyarrow conversion setup and the zstd
# codec), exceeding the default 200 ms deadline; later examples take
# well under 1 ms.
no_deadline = settings(deadline=None)


@pytest.fixture(scope="module")
def feather_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory for the feather files written by these tests.

    `to_feather` requires a real filesystem path (the destination goes
    through `os.fsdecode`), so unlike the parquet sibling this file cannot
    write to an in-memory filesystem. A module-scoped fixture keeps
    Hypothesis happy (a function-scoped one would trip
    `HealthCheck.function_scoped_fixture`); pytest-xdist workers get
    separate base directories from `tmp_path_factory` automatically.
    """
    return tmp_path_factory.mktemp("test-to-from-feather")


def feather_path(directory: Path) -> str:
    """A fixed path in `feather_dir`, overwritten by each example.

    Under pytest-run-parallel the test body runs concurrently on multiple
    threads sharing the module-scoped directory, so use a per-thread
    filename to avoid one thread reading the file while another is still
    writing it.
    """
    return str(directory / f"{threading.get_ident()}.feather")


def feather_dtype(dtype: np.dtype) -> bool:
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


SUPPORTED_FEATHER_DTYPES = tuple(d for d in SUPPORTED_DTYPES if feather_dtype(d))


def feather_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for dtypes that survive the feather roundtrip."""
    return st.sampled_from(SUPPORTED_FEATHER_DTYPES)


def feather_compatible(a: ak.Array) -> bool:
    """Arrays whose type the feather table representation can hold.

    The clauses here are limits of the format itself and stay even when
    every cited issue is fixed.
    """
    layout, is_option = _strip_options(a.layout)
    if layout.is_record:
        if layout.fields == [""]:
            # collides with the anonymous column that `to_feather` uses
            # for non-record arrays; reads back unwrapped
            return False
        if not layout.is_tuple and len(layout.fields) == 0:
            # becomes a zero-column table, losing the length
            return False
        if is_option and all(x.is_option for x in layout.contents):
            # a valid row whose fields are all null reads back as a null
            # row (outermost struct validity is not stored)
            return False
    return True


def has_issues(a: ak.Array) -> bool:
    """Arrays affected by a known issue in the conversion.

    One function per known issue, each named after the issue whose
    released fix deletes it; with all of them gone the test filters
    reduce to `feather_compatible` and `feather_stable`.
    """
    if _has_issue_4221(a.layout):
        return True
    if _has_issue_4222(a.layout):
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
    and a `ListArray` may be compacted from anywhere; such nodes can
    present any list below an option with nonzero offsets.
    """
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


def _has_issue_4229(layout: ak.contents.Content) -> bool:
    """`from_feather` loses the length of a size-0 fixed-size list (a
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


@no_deadline
@given(
    a=st_ak.constructors.arrays(
        dtypes=feather_dtypes(),
        # a bare unknown type crashes `from_feather` (AttributeError in
        # `form_remove_optiontype`, #4230), and one nested in a record
        # reads back as an option
        allow_empty=False,
        # pyarrow's IPC file format drops the buffers of a union wrapped
        # in an extension type, so `from_feather` crashes reading the
        # invalid file that `to_feather` writes (#4238)
        allow_union=False,
        allow_indexed_option=False,
        allow_byte_masked=False,
        allow_bit_masked=False,
        allow_unmasked=False,
    ).filter(lambda a: feather_compatible(a) and not has_issues(a))
)
def test_roundtrip(feather_dir, a: ak.Array) -> None:
    """`to_feather` followed by `from_feather` reconstructs the array."""
    path = feather_path(feather_dir)
    ak.to_feather(a, path)
    returned = ak.from_feather(path)
    assert ak.array_equal(a, returned, equal_nan=True)


@no_deadline
@given(
    a=st_ak.constructors.arrays(
        dtypes=feather_dtypes(),
        allow_empty=False,
        allow_union=False,
    ).filter(lambda a: feather_compatible(a) and not has_issues(a))
)
def test_roundtrip_masked(feather_dir, a: ak.Array) -> None:
    """Option-type arrays roundtrip through nullable feather columns.

    `from_feather` returns every option layout as a `BitMaskedArray` (or
    `UnmaskedArray`), so content classes are not compared.
    """
    path = feather_path(feather_dir)
    ak.to_feather(a, path)
    returned = ak.from_feather(path)
    assert ak.array_equal(a, returned, equal_nan=True, same_content_types=False)


def feather_stable(a: ak.Array) -> bool:
    """Arrays that one roundtrip fully normalizes.

    Like the clauses in `feather_compatible`, this is a limit of the
    table representation itself and stays even when every cited issue is
    fixed.
    """
    layout, _ = _strip_options(a.layout)
    if layout.is_record and layout.fields == [""]:
        inner, _ = _strip_options(layout.contents[0])
        if inner.is_record and inner.fields == [""]:
            bottom, _ = _strip_options(inner.contents[0])
            if bottom.is_record and not bottom.is_tuple:
                # each roundtrip unwraps one level of anonymous single-field
                # records (`pass_empty_field`), and converting a non-tuple
                # record spreads its fields over the table columns (a tuple
                # stays in one anonymous column); a chain of two anonymous
                # levels over such a record still differs after one roundtrip
                return False
    return True


@no_deadline
@given(
    a=st_ak.constructors.arrays(
        dtypes=feather_dtypes(),
        # pyarrow table equality counts NaN as unequal to itself and has
        # no `equal_nan` option; NaN coverage lives in the tests above
        allow_nan=False,
        # `?{x: unknown}` crashes on the first read at the #4230 site
        allow_empty=False,
        allow_union=False,
    ).filter(lambda a: feather_stable(a) and not has_issues(a))
)
def test_roundtrip_stable(feather_dir, a: ak.Array) -> None:
    """One roundtrip brings the conversion to a fixed point.

    The roundtrip can lose information without corrupting values: an
    anonymous single-field record reads back unwrapped, a zero-field
    record loses its length, and a valid all-null-fields row under an
    option reads back null. After one full roundtrip these normalizations
    have happened: converting the reconstruction again writes an identical
    Arrow table. Layouts that the conversion crashes on or corrupts
    (#4221, #4222) are still excluded: this test does not vouch for them.
    """
    path = feather_path(feather_dir)
    ak.to_feather(a, path)
    r = ak.from_feather(path)
    ak.to_feather(r, path)
    r2 = ak.from_feather(path)
    t = ak.to_arrow_table(r)
    t2 = ak.to_arrow_table(r2)
    assert t2.equals(t)
