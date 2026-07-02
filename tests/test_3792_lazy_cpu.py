# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._connect.cpu.helpers import (
    filter_lists,
    list_sizes,
    select_lists,
    transform_lists,
)
from awkward._connect.cpu.ir_nodes import (
    EmptyLike,
    Filter,
    Input,
    IRNode,
    ListSizes,
    SelectLists,
    TransformLists,
    compute,
    reset_cache,
    topological_order,
)

# ----------------------------------------------------------------------
# Generic execution machinery (backend-neutral core)
# ----------------------------------------------------------------------


class Add(IRNode):
    def lower(self, a, b):
        return a + b


def test_lazy_execution():
    x = Add(1, 2)
    assert x._value is None
    assert x.compute() == 3
    assert x._value == 3


def test_dependency_execution_order():
    a = Add(1, 2)
    b = Add(a, 3)
    c = Add(b, 4)

    assert c.compute() == 10
    assert (a._value, b._value, c._value) == (3, 6, 10)


def test_memoization():
    calls = []

    class Spy(IRNode):
        def lower(self, x):
            calls.append(1)
            return x + 1

    n = Spy(10)
    assert n.compute() == 11
    assert n.compute() == 11
    assert len(calls) == 1


def test_memoization_of_none_result():
    calls = []

    class NoneSpy(IRNode):
        def lower(self, x):
            calls.append(1)
            return None

    n = NoneSpy(10)
    assert n.compute() is None
    assert n.compute() is None
    assert len(calls) == 1


def test_reset_cache():
    a = Add(1, 2)
    b = Add(a, 3)

    assert b.compute() == 6
    reset_cache(b)
    assert a._value is None
    assert b._value is None
    assert b.compute() == 6


def test_topological_order():
    a = Add(1, 2)
    b = Add(a, 3)
    c = Add(b, 4)

    order = topological_order(c)
    assert order == [a, b, c]


def test_diamond_graph():
    a = Add(1, 2)
    b = Add(a, 10)
    c = Add(a, 20)
    d = Add(b, c)

    order = topological_order(d)
    assert order[0] is a
    assert order[-1] is d
    assert len(order) == 4
    assert d.compute() == (3 + 10) + (3 + 20)


def test_compute_helper():
    a = Add(1, 2)
    assert compute(a) == 3
    assert compute(5) == 5


# ----------------------------------------------------------------------
# CPU nodes and helpers
# ----------------------------------------------------------------------


@pytest.fixture
def simple_array():
    return ak.Array([[1, 2], [], [3, 4, 5]])


def test_list_sizes(simple_array):
    assert list_sizes(simple_array).tolist() == [2, 0, 3]
    assert ListSizes(Input(simple_array)).compute().tolist() == [2, 0, 3]


def test_filter_lists(simple_array):
    result = filter_lists(simple_array, lambda x: x > 2)
    assert ak.to_list(result) == [[], [], [3, 4, 5]]


def test_filter_node(simple_array):
    node = Filter(Input(simple_array), lambda x: x > 2)
    assert ak.to_list(node.compute()) == [[], [], [3, 4, 5]]


def test_filter_keeps_nothing(simple_array):
    result = filter_lists(simple_array, lambda x: x > 100)
    assert ak.to_list(result) == [[], [], []]


def test_select_lists(simple_array):
    result = select_lists(simple_array, np.array([True, False, True]))
    assert ak.to_list(result) == [[1, 2], [3, 4, 5]]


def test_select_lists_node(simple_array):
    node = SelectLists(Input(simple_array), np.array([False, True, True]))
    assert ak.to_list(node.compute()) == [[], [3, 4, 5]]


def test_select_lists_does_not_corrupt_input(simple_array):
    before = ak.to_list(simple_array)
    select_lists(simple_array, np.array([True, False, False]))
    assert ak.to_list(simple_array) == before


def test_select_lists_wrong_mask_length(simple_array):
    with pytest.raises(ValueError):
        select_lists(simple_array, np.array([True, False]))


def test_transform_lists():
    arr = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = np.empty(3, dtype=np.int64)
    result = transform_lists(arr, out, 3, lambda x, y, z: x + y + z)
    assert result.tolist() == [6, 15, 24]


def test_transform_lists_node():
    arr = ak.Array([[1, 2], [3, 4]])
    out = np.empty(2, dtype=np.int64)
    node = TransformLists(Input(arr), out, 2, lambda x, y: x * y)
    assert node.compute().tolist() == [2, 12]


def test_empty_like_node(simple_array):
    result = EmptyLike(Input(simple_array)).compute()
    assert isinstance(result, ak.Array)
    assert len(result) == len(simple_array)
    # Offsets are copied, not shared
    assert (
        result.layout.offsets.data.ctypes.data
        != simple_array.layout.offsets.data.ctypes.data
    )


def test_non_list_input_raises():
    with pytest.raises(TypeError):
        list_sizes(ak.Array([1, 2, 3]))


# ----------------------------------------------------------------------
# LazyAwkwardArray on the CPU backend
# ----------------------------------------------------------------------


@pytest.fixture
def arr():
    return ak.Array([[1, 2, 3], [4, 5], [6, 7, 8, 9]])


def test_lazy_arithmetic(arr):
    lazy_arr = ak.cpu.lazy(arr)
    transformed = lazy_arr * 2 + 1
    assert ak.to_list(transformed.compute()) == [
        [3, 5, 7],
        [9, 11],
        [13, 15, 17, 19],
    ]


def test_lazy_filter(arr):
    lazy_arr = ak.cpu.lazy(arr)
    transformed = lazy_arr * 2 + 1
    result = transformed.filter(lazy_arr > 3)
    assert ak.to_list(result.compute()) == [[], [9, 11], [13, 15, 17, 19]]


def test_lazy_filter_on_transformed(arr):
    lazy_arr = ak.cpu.lazy(arr)
    transformed = lazy_arr * 2 + 1
    result = transformed.filter(transformed > 5)
    assert ak.to_list(result.compute()) == [[7], [9, 11], [13, 15, 17, 19]]


def test_lazy_combinations():
    arr2 = ak.Array([[1, 2, 3], [4, 5]])
    pairs = ak.cpu.lazy(arr2).combinations(2)
    assert ak.to_list(pairs.compute()) == [[(1, 2), (1, 3), (2, 3)], [(4, 5)]]


def test_lazy_field_access_on_combinations():
    arr3 = ak.Array([[1, 2, 3], [4, 5]])
    pairs = ak.cpu.lazy(arr3).combinations(2)
    pair_sums = pairs["0"] + pairs["1"]
    assert ak.to_list(pair_sums.compute()) == [[3, 4, 5], [9]]


def test_lazy_sum(arr):
    result = ak.cpu.lazy(arr).sum()
    assert ak.to_list(result.compute()) == [6, 9, 30]


def test_lazy_is_actually_lazy(arr):
    lazy_arr = ak.cpu.lazy(arr)
    transformed = lazy_arr * 2 + 1
    # Nothing computed yet
    assert transformed._computed_result is None


def test_compute_is_memoized(arr):
    lazy_arr = ak.cpu.lazy(arr)
    result = (lazy_arr * 2 + 1).filter(lazy_arr > 3)
    assert result.compute() is result.compute()


def test_invalidate(arr):
    lazy_arr = ak.cpu.lazy(arr)
    result = lazy_arr * 2
    first = result.compute()
    result.invalidate()
    second = result.compute()
    assert first is not second
    assert ak.to_list(first) == ak.to_list(second)


def test_memo_skips_leaves(arr):
    from awkward._connect.lazy._executor import IRExecutor
    from awkward._connect.lazy._lazy_impl import LazyAwkwardArray

    ex = IRExecutor()
    la = LazyAwkwardArray.from_array(arr, ex)
    (la * 2 + 1).compute()
    # Only the mul and add nodes are cached; InputNode/ConstantNode are not
    assert len(ex._memo) == 2


def test_memo_lru_eviction_by_entries(arr):
    from awkward._connect.lazy._executor import IRExecutor
    from awkward._connect.lazy._lazy_impl import LazyAwkwardArray

    ex = IRExecutor(max_entries=2, max_bytes=None)
    la = LazyAwkwardArray.from_array(arr, ex)
    shared = la * 2

    r1 = shared + 1
    r1.compute()
    assert len(ex._memo) == 2  # shared, r1

    # Computing r2 hits `shared` (refreshing it) and evicts r1, the LRU entry
    r2 = shared + 2
    r2.compute()
    assert len(ex._memo) == 2
    assert shared.ir_node.node_id in ex._memo
    assert r2.ir_node.node_id in ex._memo
    assert r1.ir_node.node_id not in ex._memo


def test_memo_eviction_by_bytes(arr):
    from awkward._connect.lazy._executor import IRExecutor
    from awkward._connect.lazy._lazy_impl import LazyAwkwardArray

    one_result = (arr * 2).layout.nbytes

    # Budget fits one result (plus slack) but not two
    ex = IRExecutor(max_entries=None, max_bytes=int(one_result * 1.5))
    la = LazyAwkwardArray.from_array(arr, ex)
    (la * 2).compute()
    (la * 3).compute()
    assert len(ex._memo) == 1
    assert ex._memo_bytes <= ex.max_bytes


def test_memo_oversized_result_not_cached(arr):
    from awkward._connect.lazy._executor import IRExecutor
    from awkward._connect.lazy._lazy_impl import LazyAwkwardArray

    ex = IRExecutor(max_entries=None, max_bytes=1)
    la = LazyAwkwardArray.from_array(arr, ex)
    result = (la * 2).compute()
    assert len(ex._memo) == 0
    assert ex._memo_bytes == 0
    # The wrapper still caches its own root result
    assert ak.to_list(result) == ak.to_list(arr * 2)


def test_memo_invalidate_resets_byte_accounting(arr):
    from awkward._connect.lazy._executor import IRExecutor
    from awkward._connect.lazy._lazy_impl import LazyAwkwardArray

    ex = IRExecutor()
    la = LazyAwkwardArray.from_array(arr, ex)
    (la * 2).compute()
    assert ex._memo_bytes > 0
    ex.invalidate()
    assert len(ex._memo) == 0
    assert len(ex._memo_sizes) == 0
    assert ex._memo_bytes == 0


def test_memo_unbounded_when_limits_none(arr):
    from awkward._connect.lazy._executor import IRExecutor
    from awkward._connect.lazy._lazy_impl import LazyAwkwardArray

    ex = IRExecutor(max_entries=None, max_bytes=None)
    la = LazyAwkwardArray.from_array(arr, ex)
    for i in range(10):
        (la + i).compute()
    assert len(ex._memo) == 10


def test_bool_raises(arr):
    lazy_arr = ak.cpu.lazy(arr)
    with pytest.raises(TypeError):
        bool(lazy_arr == lazy_arr)


def test_visualize(arr):
    lazy_arr = ak.cpu.lazy(arr)
    text = ((lazy_arr * 2 + 1).filter(lazy_arr > 3)).visualize()
    assert isinstance(text, str)
    assert "filter" in text


def test_cuda_backed_array_is_rejected(arr):
    # ak.cpu.lazy must reject non-cpu arrays; forging the check is enough here
    class FakeBackend:
        name = "cuda"

    class FakeLayout:
        backend = FakeBackend()

    class FakeArray:
        layout = FakeLayout()

    with pytest.raises(TypeError):
        ak.cpu.lazy(FakeArray())
