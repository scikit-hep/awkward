# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import threading

import hypothesis_awkward.strategies as st_ak
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import awkward as ak
from tests.properties.operations import util

pytest.importorskip("pyarrow.parquet")

# The first example pays one-time lazy imports (pyarrow.parquet, fsspec's
# memory filesystem) of ~350 ms, exceeding hypothesis' default 200 ms
# deadline; later examples take ~1 ms.
no_deadline = settings(deadline=None)


def parquet_path() -> str:
    """A fixed path in fsspec's in-memory filesystem, overwritten by each
    example.

    Under pytest-run-parallel the test body runs concurrently on multiple
    threads sharing the process-global in-memory filesystem, so use a
    per-thread filename to avoid one thread reading the file while another
    is still writing it.
    """
    return f"memory://test-to-from-parquet/{threading.get_ident()}.parquet"


def parquet_dtype(dtype: np.dtype) -> bool:
    if dtype.kind == "c":
        # pyarrow cannot write complex numbers (ArrowNotImplementedError)
        return False
    if dtype.kind == "f" and dtype.itemsize == 16:
        # pyarrow cannot write extended-precision floats, which exist on
        # some platforms (e.g. float128 on Linux)
        return False
    if dtype.kind == "M":
        # Only "ns" roundtrips: coarser units are unsupported or coerced
        # ("s" -> "ms"; "D" corrupts values, #4219), and out-of-range
        # values such as NaT in "ms"/"us" crash `from_parquet` reading
        # row-group statistics (OverflowError, #4220).
        return np.datetime_data(dtype)[0] == "ns"
    if dtype.kind == "m":
        # pyarrow durations support only seconds and finer units
        return np.datetime_data(dtype)[0] in {"s", "ms", "us", "ns"}
    return True


def parquet_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for dtypes that survive the parquet roundtrip."""
    return st_ak.numpy.supported_dtypes().filter(parquet_dtype)


def parquet_writable(a: ak.Array) -> bool:
    layout, is_option = a.layout, False
    while layout.is_option or layout.is_indexed:
        is_option = is_option or layout.is_option
        layout = layout.content
    if layout.is_record:
        if layout.fields == [""]:
            # collides with the anonymous column that `to_parquet` uses
            # for non-record arrays; reads back unwrapped
            return False
        if is_option and all(x.is_option for x in layout.contents):
            # a valid row whose fields are all null reads back as a null
            # row (outermost struct validity is not stored)
            return False
    return (
        _nodes_writable(a.layout)
        and not _has_issue_4554(a.layout)
        and not _has_issue_4555(a.layout)
    )


def _nodes_writable(layout: ak.contents.Content, nullable: bool = False) -> bool:
    """Exclude layouts that `to_arrow` currently mishandles.

    `nullable` tracks whether Arrow validity bytes flow into this node from
    an enclosing option: they start at option nodes, pass through records
    and indexed nodes, and stop below lists. A var-length list receiving
    validity bytes whose offsets do not start at zero is compacted against
    unshifted content in `ListOffsetArray._to_arrow`, shifting every list
    by `offsets[0]` (data corruption, #4222); indexed nodes can recreate such
    offsets from anywhere in their content, so they are excluded outright
    when nullable.
    """


def _has_issue_4305(layout: ak.contents.Content, nullable: bool = False) -> bool:
    """Return `True` if an `UnmaskedArray` receives validity from an
    enclosing option.

    `to_arrow` records an `UnmaskedArray`'s class name as `mask_type` on
    the content it wraps, noting that the node carried no mask, and parquet
    stores each column's validity in full, so an enclosing option's nulls
    arrive in that content's validity bitmap. `from_parquet` requires a
    node with that metadata to read back all-ones (message-less
    `AssertionError` in `popbuffers_finalize`, #4305).

    `nullable` tracks whether Arrow validity bytes flow into this node from
    an enclosing option: they start at option nodes, pass through records
    and indexed nodes, and stop below lists.

    The condition here is wider than the failure in three ways, each
    chosen over a special case: an option holding no null roundtrips; an
    array shorter than eight roundtrips, because the assertion compares
    whole bytes (`[: len(out) // 8]`) and so compares nothing; and the one
    root-level shape `option(record(...))` roundtrips, because
    `to_parquet` projects the enclosing option into each field of a
    root-level named record, replacing the `UnmaskedArray` with an
    `IndexedOptionArray`. Every other root-level option shape fails.
    """
    if layout.is_record:
        return any(_has_issue_4305(x, nullable) for x in layout.contents)
    if layout.is_regular:
        return _has_issue_4305(layout.content, False)
    if layout.is_option:
        if nullable and isinstance(layout, ak.contents.UnmaskedArray):
            return True
        return _has_issue_4305(layout.content, True)
    if layout.is_indexed:
        return _has_issue_4305(layout.content, nullable)
    if layout.is_list:
        return _has_issue_4305(layout.content, False)
    return False


def _nodes_writable(layout: ak.contents.Content) -> bool:
    """Exclude layouts that `to_arrow` currently mishandles."""
    if layout.is_record:
        if len(layout.fields) == 0:
            # named records lose their length; tuples cannot be written
            # (pyarrow: "Cannot write struct type '' with no child field")
            return False
        return all(_nodes_writable(x) for x in layout.contents)
    if layout.is_regular:
        # `to_arrow` drops the length of a size-0 `RegularArray`
        # (pyarrow: "Expected all lists to be of size=0")
        return layout.size > 0 and _nodes_writable(layout.content)
    if layout.is_option:
        if isinstance(layout, ak.contents.IndexedOptionArray):
            if layout.length > 0 and layout.content.length == 0:
                # `to_arrow` crashes projecting nulls over a zero-length
                # content (IndexError in `ListOffsetArray._to_arrow`, #4221)
                return False
        return _nodes_writable(layout.content)
    if layout.is_indexed:
        return _nodes_writable(layout.content)
    if layout.is_list:
        return _nodes_writable(layout.content)
    return True


def _has_issue_4554(layout: ak.contents.Content, under_option: bool = False) -> bool:
    """`from_arrow` raises `TypeError` on an option-type categorical
    array: with the default `categorical_as_dictionary=False`, `to_arrow`
    writes the projected values but keeps `__array__: "categorical"` in
    the extension type's `mask_parameters`, and the `BitMaskedArray` (or
    `UnmaskedArray`) that `from_arrow` builds with them rejects it
    (#4554). A table conversion moves an option over a record into each
    field, so a categorical node below any option is affected too.

    `under_option` tracks whether an option node encloses this node.
    """
    if layout.parameter("__array__") == "categorical" and (
        under_option or layout.is_option
    ):
        return True
    under_option = under_option or layout.is_option
    return any(_has_issue_4554(x, under_option) for x in _children(layout))


def _has_issue_4555(layout: ak.contents.Content) -> bool:
    """`to_arrow` raises `TypeError` on an indexed node with parameters
    over a zero-length content: `IndexedArray._to_arrow` then converts
    only the content and copies the node's parameters onto it, and the
    content rejects them (an `EmptyArray` accepts no parameters, and a
    `ListOffsetArray` or `NumpyArray` rejects `__array__: "categorical"`,
    so every categorical array with no categories fails; #4555)."""
    if layout.is_indexed and layout.content.length == 0 and layout.parameters:
        return True
    return any(_has_issue_4555(x) for x in _children(layout))


def _children(layout: ak.contents.Content) -> list[ak.contents.Content]:
    """The direct child layouts of a node."""
    if layout.is_record or layout.is_union:
        return list(layout.contents)
    if layout.is_option or layout.is_indexed or layout.is_list:
        return [layout.content]
    return []


@no_deadline
@given(
    a=st_ak.constructors.arrays(
        dtypes=parquet_dtypes(),
        # a bare unknown type cannot be written ("A null type field may
        # not be non-nullable"), and one nested in a record or regular
        # list reads back as an option
        allow_empty=False,
        # pyarrow cannot write unions to parquet
        allow_union=False,
        allow_indexed_option=False,
        allow_byte_masked=False,
        allow_bit_masked=False,
        allow_unmasked=False,
    ).filter(parquet_writable)
)
def test_roundtrip(a: ak.Array) -> None:
    """`to_parquet` followed by `from_parquet` reconstructs the array."""
    path = parquet_path()
    ak.to_parquet(a, path)
    returned = ak.from_parquet(path)
    assert ak.array_equal(util.without_categorical(a), returned, equal_nan=True)


@no_deadline
@given(
    a=st_ak.constructors.arrays(
        dtypes=parquet_dtypes(),
        allow_empty=False,
        # an option directly over a `RegularArray` cannot be written
        # (ARROW-14547: "Lists with non-zero length null components are
        # not supported")
        allow_regular=False,
        allow_union=False,
    ).filter(parquet_writable)
)
def test_roundtrip_masked(a: ak.Array) -> None:
    """Option-type arrays roundtrip through nullable parquet columns.

    `from_parquet` returns every option layout as a `BitMaskedArray` (or
    `UnmaskedArray`), so content classes are not compared.
    """
    path = parquet_path()
    ak.to_parquet(a, path)
    returned = ak.from_parquet(path)
    assert ak.array_equal(
        util.without_categorical(a), returned, equal_nan=True, same_content_types=False
    )


@no_deadline
@given(
    attrs=st.dictionaries(
        # keys starting with "@" are transient and intentionally not
        # written to parquet
        st.text().filter(lambda k: not k.startswith("@")),
        st.none()
        | st.booleans()
        | st.integers()
        | st.floats(allow_nan=False)
        | st.text(),
    )
)
def test_roundtrip_attrs(attrs: dict) -> None:
    """`attrs` survive the parquet roundtrip."""
    a = ak.Array([1.1, 2.2, 3.3], attrs=attrs)
    path = parquet_path()
    ak.to_parquet(a, path)
    assert ak.from_parquet(path).attrs == attrs
