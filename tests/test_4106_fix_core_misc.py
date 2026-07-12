# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward._categorical import as_hashable
from awkward._do import recursively_apply
from awkward._errors import OperationErrorContext
from awkward.prettyprint import Formatter


def test_hashablelist_with_dict_elements_is_hashable():
    """HashableList with dict elements used to raise TypeError: unhashable type 'dict'."""
    obj = [{"x": 1}, {"x": 2}]
    h = as_hashable(obj)
    # Must be hashable and usable as a dict key
    lookup = {h: "found"}
    assert lookup[as_hashable(obj)] == "found"


def test_categorical_equal_list_of_lists():
    """Comparing categorical arrays whose categories are lists (different orders)."""
    categories1 = ak.Array([[1, 2], [3, 4]])
    categories2 = ak.Array([[3, 4], [1, 2]])  # reversed order
    index1 = ak.index.Index64(np.array([0, 1, 0], dtype=np.int64))
    index2 = ak.index.Index64(np.array([0, 0, 1], dtype=np.int64))
    cat1 = ak.contents.IndexedArray(
        index1, categories1.layout, parameters={"__array__": "categorical"}
    )
    cat2 = ak.contents.IndexedArray(
        index2, categories2.layout, parameters={"__array__": "categorical"}
    )
    # cat1 values: [1,2] [3,4] [1,2]; cat2 values: [3,4] [3,4] [1,2]
    result = (ak.Array(cat1) == ak.Array(cat2)).to_list()
    assert result == [False, True, True]


def test_any_backend_is_delayed_continues_after_unknown_object():
    """A plain (non-array) item before an eager array must not prevent detection.

    Before the fix, encountering an unrecognised object caused an unconditional
    return False, so any arrays later in the argument list were never checked.
    """
    ctx = OperationErrorContext.__new__(OperationErrorContext)
    eager = ak.Array([1, 2, 3])
    assert ctx.any_backend_is_delayed([42, eager]) is False


def test_any_backend_is_delayed_nested_non_delayed_continues():
    """Nested iterable returning False from recursion must not short-circuit outer loop."""
    ctx = OperationErrorContext.__new__(OperationErrorContext)
    eager = ak.Array([1, 2])
    result = ctx.any_backend_is_delayed([[42], eager], depth_limit=2)
    assert result is False


def _make_custom_str_array(n=5):
    class MyPoint(ak.Record):
        def __str__(self):
            return "POINT"

    return ak.Array(
        [{"x": i} for i in range(n)],
        behavior={"MyPoint": MyPoint},
        with_name="MyPoint",
    )


def test_prettyprint_custom_str_not_spliced():
    """custom_str result must be treated as a single token, not spliced char-by-char."""
    from awkward.prettyprint import valuestr_horiz

    arr = _make_custom_str_array(5)
    cols_taken, strs = valuestr_horiz(arr, 80, Formatter())

    # Before the fix, front.extend(str) iterated characters; "POINT" would appear
    # as individual letters.  Now each token must be "POINT" in full.
    single_letters_from_point = set("POINT")
    for s in strs:
        if len(s) == 1 and s in single_letters_from_point:
            raise AssertionError(
                f"custom_str was spliced char-by-char: found {s!r} in {strs}"
            )
    # cols_taken must match the actual rendered width
    joined = "".join(strs)
    assert len(joined) == cols_taken


def test_prettyprint_repr_does_not_explode():
    """repr() of an array with custom-str records must not raise or overflow."""
    arr = _make_custom_str_array(5)
    result = repr(arr)
    assert isinstance(result, str)
    assert len(result) < 500


def test_recursively_apply_record_regular_to_jagged_forwarded():
    """recursively_apply with regular_to_jagged=True on a Record must forward the arg."""
    from awkward.contents.recordarray import RecordArray
    from awkward.record import Record as LowRecord

    regular = ak.to_regular(ak.Array([[1, 2], [3, 4]]))
    ra = RecordArray([regular.layout], ["vals"])
    low_rec = LowRecord(ra, 0)

    seen_options = []

    def action(layout, **kwargs):
        seen_options.append(kwargs.get("options", {}))
        return None

    recursively_apply(low_rec, action, regular_to_jagged=True)
    assert any(opts.get("regular_to_jagged") is True for opts in seen_options)


def test_recursively_apply_record_regular_to_jagged_converts_type():
    """With regular_to_jagged=True, RegularArray inside a Record is converted."""
    from awkward.contents.recordarray import RecordArray
    from awkward.record import Record as LowRecord

    regular = ak.to_regular(ak.Array([[1, 2], [3, 4]]))
    ra = RecordArray([regular.layout], ["vals"])
    low_rec = LowRecord(ra, 0)

    seen_types = []

    def action(layout, **kwargs):
        seen_types.append(type(layout).__name__)
        return None

    recursively_apply(low_rec, action, regular_to_jagged=True)

    # RegularArray should be converted to ListOffsetArray
    assert "RegularArray" not in seen_types, f"seen: {seen_types}"
    assert "ListOffsetArray" in seen_types
