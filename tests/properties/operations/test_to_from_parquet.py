# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import threading

import hypothesis_awkward.strategies as st_ak
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis_awkward.util.dtype import SUPPORTED_DTYPES

import awkward as ak

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


SUPPORTED_PARQUET_DTYPES = tuple(d for d in SUPPORTED_DTYPES if parquet_dtype(d))


def parquet_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for dtypes that survive the parquet roundtrip."""
    return st.sampled_from(SUPPORTED_PARQUET_DTYPES)


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
    return _nodes_writable(a.layout)


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
    if layout.is_record:
        if len(layout.fields) == 0:
            # named records lose their length; tuples cannot be written
            # (pyarrow: "Cannot write struct type '' with no child field")
            return False
        return all(_nodes_writable(x, nullable) for x in layout.contents)
    if layout.is_regular:
        # `to_arrow` drops the length of a size-0 `RegularArray`
        # (pyarrow: "Expected all lists to be of size=0")
        return layout.size > 0 and _nodes_writable(layout.content, False)
    if layout.is_option:
        if isinstance(layout, ak.contents.IndexedOptionArray):
            if layout.length > 0 and layout.content.length == 0:
                # `to_arrow` crashes projecting nulls over a zero-length
                # content (IndexError in `ListOffsetArray._to_arrow`, #4221)
                return False
            if nullable and layout.content.is_list:
                return False
        return _nodes_writable(layout.content, True)
    if layout.is_indexed:
        if nullable and layout.content.is_list:
            return False
        return _nodes_writable(layout.content, nullable)
    if layout.is_list:
        if nullable and layout.length > 0 and layout.starts[0] != 0:
            return False
        return _nodes_writable(layout.content, False)
    return True


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
    assert ak.array_equal(a, returned, equal_nan=True)


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
    assert ak.array_equal(a, returned, equal_nan=True, same_content_types=False)


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
