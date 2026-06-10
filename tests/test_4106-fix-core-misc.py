# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._categorical import HashableDict, HashableList, as_hashable
from awkward._do import recursively_apply
from awkward._errors import OperationErrorContext
from awkward.prettyprint import Formatter


# ---------------------------------------------------------------------------
# Fix 1: HashableList must recurse through as_hashable
# ---------------------------------------------------------------------------


def test_hashablelist_recurses_for_dicts():
    """HashableList should apply as_hashable to each element (like HashableDict)."""
    # list containing dicts — used to raise TypeError: unhashable type 'dict'
    obj = [{"x": 1}, {"x": 2}]
    h = as_hashable(obj)
    assert isinstance(h, HashableList)
    # Each element should have been converted to a HashableDict, not a raw dict
    for v in h.values:
        assert isinstance(v, HashableDict)
    # Must be hashable
    assert isinstance(hash(h), int)


def test_hashablelist_recurses_for_nested_lists():
    """Nested lists of lists should also be recursively hashable."""
    obj = [[1, 2], [3, 4]]
    h = as_hashable(obj)
    assert isinstance(h, HashableList)
    for v in h.values:
        assert isinstance(v, HashableList)
    assert isinstance(hash(h), int)


def test_hashablelist_can_be_dict_key():
    """HashableList containing dicts must be usable as a dict key (was broken)."""
    obj1 = [{"x": 1}, {"x": 2}]
    obj2 = [{"x": 1}, {"x": 2}]
    obj3 = [{"x": 3}]
    h1 = as_hashable(obj1)
    h2 = as_hashable(obj2)
    h3 = as_hashable(obj3)
    lookup = {h1: "a", h3: "b"}
    assert lookup[h2] == "a"  # h2 equals h1
    assert h1 == h2
    assert h1 != h3


def test_categorical_equal_list_of_lists():
    """Comparing categorical arrays whose categories are lists of lists (different orders)."""
    # Different category content objects forces the hash-based mapping path
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
    # equal: F T T
    result = (ak.Array(cat1) == ak.Array(cat2)).to_list()
    assert result == [False, True, True]


def test_categorical_equal_nested_list_of_dicts_hashable():
    """HashableList with dict elements: as_hashable on a list-of-dict must not raise."""
    items = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    h1 = as_hashable(items)
    h2 = as_hashable(items)
    assert isinstance(hash(h1), int)
    assert h1 == h2
    # Can be used as dict key
    lookup = {h1: 0}
    assert lookup[h2] == 0


# ---------------------------------------------------------------------------
# Fix 2: any_backend_is_delayed must not short-circuit on non-delayed items
# ---------------------------------------------------------------------------


def test_any_backend_is_delayed_continues_after_unknown_object():
    """A plain (non-array) item before an eager array must not prevent detection."""
    ctx = OperationErrorContext.__new__(OperationErrorContext)
    # A plain Python scalar has no backend; an eager array follows it.
    # The loop must not return False at the scalar and miss the array.
    eager = ak.Array([1, 2, 3])
    assert ctx.any_backend_is_delayed([42, eager]) is False


def test_any_backend_is_delayed_all_plain_returns_false():
    """All-plain-python iterable returns False without crashing."""
    ctx = OperationErrorContext.__new__(OperationErrorContext)
    assert ctx.any_backend_is_delayed([1, "hello", object()]) is False


def test_any_backend_is_delayed_two_eager_arrays():
    """Two eager arrays both return False; loop must complete."""
    ctx = OperationErrorContext.__new__(OperationErrorContext)
    eager1 = ak.Array([1, 2, 3])
    eager2 = ak.Array([4, 5, 6])
    assert ctx.any_backend_is_delayed([eager1, eager2]) is False


def test_any_backend_is_delayed_nested_non_delayed_continues():
    """Nested iterable returning False from recursion doesn't short-circuit outer loop."""
    ctx = OperationErrorContext.__new__(OperationErrorContext)
    eager = ak.Array([1, 2])
    # Outer list: [plain_list, eager_array] — first item recurses and returns False,
    # but must not cause the outer loop to return False before seeing eager_array.
    result = ctx.any_backend_is_delayed([[42], eager], depth_limit=2)
    assert result is False


# ---------------------------------------------------------------------------
# Fix 3: prettyprint custom_str wrapped in list + width recomputed
# ---------------------------------------------------------------------------


def _make_custom_str_array(n=5):
    """Build an Array whose elements have custom __str__."""

    class MyPoint(ak.Record):
        def __str__(self):
            return "POINT"

    return ak.Array(
        [{"x": i} for i in range(n)],
        behavior={"MyPoint": MyPoint},
        with_name="MyPoint",
    )


def _make_custom_repr_array(n=5):
    """Build an Array whose elements have custom __repr__."""

    class MyPoint(ak.Record):
        def __repr__(self):
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

    # Each element in strs should be a multi-character token (not a single char)
    for s in strs:
        assert isinstance(s, str)
    # Tokens like "[", "POINT", ", ", "]" — none should be a single letter from "POINT"
    single_letters_from_point = set("POINT")
    for s in strs:
        if len(s) == 1 and s in single_letters_from_point:
            # A single-char from the custom string means it was spliced
            pytest.fail(
                f"custom_str was spliced char-by-char: found single char {s!r} in strs={strs}"
            )


def test_prettyprint_custom_str_width_accurate():
    """cols_taken must reflect the actual length of the custom string."""
    from awkward.prettyprint import valuestr_horiz

    arr = _make_custom_str_array(1)
    cols_taken, strs = valuestr_horiz(arr, 80, Formatter())

    # With 1 element: "[POINT]" -> cols_taken should be 7, not based on old rendering
    joined = "".join(strs)
    assert len(joined) == cols_taken, (
        f"cols_taken={cols_taken} but joined={joined!r} has len={len(joined)}"
    )


def test_prettyprint_repr_does_not_explode():
    """repr() of an array with custom-str records must not raise or overflow."""
    arr = _make_custom_str_array(5)
    result = repr(arr)
    assert isinstance(result, str)
    # Should not explode into thousands of chars
    assert len(result) < 500


def test_prettyprint_repr_custom_repr_does_not_explode():
    """repr() of an array with custom-repr records must not raise or overflow."""
    arr = _make_custom_repr_array(5)
    result = repr(arr)
    assert isinstance(result, str)
    assert len(result) < 500


# ---------------------------------------------------------------------------
# Fix 4: recursively_apply on Record forwards regular_to_jagged
# ---------------------------------------------------------------------------


def _make_record_with_regular_content():
    """Build a low-level Record whose array contains a RegularArray."""
    from awkward.contents.recordarray import RecordArray
    from awkward.record import Record as LowRecord

    regular = ak.to_regular(ak.Array([[1, 2], [3, 4]]))  # RegularArray
    ra = RecordArray([regular.layout], ["vals"])
    return LowRecord(ra, 0)


def test_recursively_apply_record_regular_to_jagged_forwarded():
    """recursively_apply with regular_to_jagged=True on a Record must forward the arg."""
    low_rec = _make_record_with_regular_content()

    seen_options = []

    def action(layout, **kwargs):
        seen_options.append(kwargs.get("options", {}))
        return None

    recursively_apply(low_rec, action, regular_to_jagged=True)
    assert any(opts.get("regular_to_jagged") is True for opts in seen_options)


def test_recursively_apply_record_regular_to_jagged_false_forwarded():
    """regular_to_jagged=False is also forwarded correctly."""
    low_rec = _make_record_with_regular_content()

    seen_options = []

    def action(layout, **kwargs):
        seen_options.append(kwargs.get("options", {}))
        return None

    recursively_apply(low_rec, action, regular_to_jagged=False)
    assert any(opts.get("regular_to_jagged") is False for opts in seen_options)


def test_recursively_apply_record_regular_to_jagged_converts_type():
    """With regular_to_jagged=True, RegularArray inside a Record is converted."""
    from awkward.contents.recordarray import RecordArray
    from awkward.record import Record as LowRecord

    regular = ak.to_regular(ak.Array([[1, 2], [3, 4]]))  # RegularArray
    ra = RecordArray([regular.layout], ["vals"])
    low_rec = LowRecord(ra, 0)

    seen_types = []

    def action(layout, **kwargs):
        seen_types.append(type(layout).__name__)
        return None

    recursively_apply(low_rec, action, regular_to_jagged=True)

    # With regular_to_jagged=True, RegularArray should be converted to ListOffsetArray
    assert "RegularArray" not in seen_types, (
        f"RegularArray was NOT converted; seen: {seen_types}"
    )
    assert "ListOffsetArray" in seen_types
