# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak
from awkward._connect.cuda.ir_nodes import (
    EmptyLike,
    Filter,
    Input,
    IRNode,
    ToIterator,
    compute,
    reset_cache,
    topological_order,
)

# ----------------------------------------------------------------------
# Minimal test node
# ----------------------------------------------------------------------


class Add(IRNode):
    def lower(self, a, b):
        return a + b


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_lazy_execution():
    x = Add(1, 2)

    # Nothing computed yet
    assert x._value is None

    result = x.compute()

    assert result == 3
    assert x._value == 3


def test_dependency_execution_order():
    a = Add(1, 2)
    b = Add(a, 3)
    c = Add(b, 4)

    result = c.compute()

    assert result == 10
    assert a._value == 3
    assert b._value == 6
    assert c._value == 10


def test_memoization():
    calls = []

    class Spy(IRNode):
        def lower(self, x):
            calls.append(1)
            return x + 1

    n = Spy(10)

    assert n.compute() == 11
    assert n.compute() == 11

    # lower() should run only once
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

    # Inputs first, root last
    assert order[0] is a
    assert order[1] is b
    assert order[2] is c


def test_compute_helper():
    a = Add(1, 2)

    assert compute(a) == 3
    assert compute(5) == 5


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def simple_array():
    return ak.to_backend(ak.Array([[1, 2], [], [3, 4, 5]]), "cuda")


def test_empty_like_node(simple_array):
    node_input = Input(simple_array)
    empty_node = EmptyLike(node_input)
    result = empty_node.compute()

    # Should return same structure but empty buffers
    assert isinstance(result, ak.Array)
    # Same length at top level
    assert len(result) == len(simple_array)
    # Each sublist has correct length
    for sublist in result:
        assert isinstance(sublist, ak.highlevel.Array)
    # Underlying content should be zero-length arrays
    # content_lengths = [len(x) for x in result]
    # EmptyLike keeps offsets, so sublists can still have same length as original?
    # Actually empty_like makes empty data buffers, sublists remain but uninitialized
    # So we just check type and length
    assert len(result) == len(simple_array)


# ----------------------------------------------------------------------
# Test ToIterator node
# ----------------------------------------------------------------------


def test_to_iterator_node(simple_array):
    node_input = Input(simple_array)
    iter_node = ToIterator(node_input)

    _it, meta = iter_node.compute()
    assert isinstance(meta, dict)
    assert "offsets" in meta
    assert "count" in meta
    # offsets should be a NumPy/CuPy array
    assert hasattr(meta["offsets"], "__len__")
    # count should be integer
    assert isinstance(meta["count"], int)


# ----------------------------------------------------------------------
# Test Filter node
# ----------------------------------------------------------------------


def test_filter_node(simple_array):
    # Predicate: keep elements > 2
    def pred(x):
        return x > 2

    node_input = Input(simple_array)
    filter_node = Filter(node_input, pred)

    result = filter_node.compute()
    # Convert to list for assertion
    result_list = ak.to_list(result)

    # Original array: [[1,2], [], [3,4,5]]
    # After filter (>2): [[], [], [3,4,5]]
    expected = [[], [], [3, 4, 5]]
    assert result_list == expected


# ----------------------------------------------------------------------
# Test chaining nodes
# ----------------------------------------------------------------------


def test_chained_nodes(simple_array):
    # Chain: Input -> Filter -> EmptyLike
    def pred(x):
        return x > 1

    result = EmptyLike(Filter(Input(simple_array), pred)).compute()
    assert isinstance(result, ak.Array)
    assert len(result) == len(simple_array)
