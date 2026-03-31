# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_concatenate():
    # Plain lists
    arrays = [[i] for i in range(5)]
    expected_list = ak.concatenate(arrays)
    expected_tuple = ak.concatenate(tuple(arrays))
    result = ak.concatenate([i] for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # ak.Array instances
    arrays = [ak.Array([i, i + 1]) for i in range(5)]
    expected_list = ak.concatenate(arrays)
    expected_tuple = ak.concatenate(tuple(arrays))
    result = ak.concatenate(ak.Array([i, i + 1]) for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # Nested arrays with axis=1
    arrays = [ak.Array([[i, i + 1], [i + 2]]) for i in range(5)]
    expected_list = ak.concatenate(arrays, axis=1)
    expected_tuple = ak.concatenate(tuple(arrays), axis=1)
    result = ak.concatenate((ak.Array([[i, i + 1], [i + 2]]) for i in range(5)), axis=1)
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)


def test_zip():
    # Plain lists
    arrays = [[i, i + 1, i + 2] for i in range(5)]
    expected_list = ak.zip(arrays)
    expected_tuple = ak.zip(tuple(arrays))
    result = ak.zip([i, i + 1, i + 2] for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # ak.Array instances
    arrays = [ak.Array([i, i + 1, i + 2]) for i in range(5)]
    expected_list = ak.zip(arrays)
    expected_tuple = ak.zip(tuple(arrays))
    result = ak.zip(ak.Array([i, i + 1, i + 2]) for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # Nested arrays
    arrays = [ak.Array([[i, i + 1], [i + 2]]) for i in range(5)]
    expected_list = ak.zip(arrays)
    expected_tuple = ak.zip(tuple(arrays))
    result = ak.zip(ak.Array([[i, i + 1], [i + 2]]) for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)


def test_zip_no_broadcast():
    # Plain lists
    arrays = [[i, i + 1, i + 2] for i in range(5)]
    expected_list = ak.zip_no_broadcast(arrays)
    expected_tuple = ak.zip_no_broadcast(tuple(arrays))
    result = ak.zip_no_broadcast([i, i + 1, i + 2] for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # ak.Array instances
    arrays = [ak.Array([i, i + 1, i + 2]) for i in range(5)]
    expected_list = ak.zip_no_broadcast(arrays)
    expected_tuple = ak.zip_no_broadcast(tuple(arrays))
    result = ak.zip_no_broadcast(ak.Array([i, i + 1, i + 2]) for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # Nested arrays
    arrays = [ak.Array([[i, i + 1], [i + 2]]) for i in range(5)]
    expected_list = ak.zip_no_broadcast(arrays)
    expected_tuple = ak.zip_no_broadcast(tuple(arrays))
    result = ak.zip_no_broadcast(ak.Array([[i, i + 1], [i + 2]]) for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)


def test_cartesian():
    # Plain nested lists, default axis=1
    arrays = [[[i, i + 1], [i + 2]] for i in range(5)]
    expected_list = ak.cartesian(arrays)
    expected_tuple = ak.cartesian(tuple(arrays))
    result = ak.cartesian([[i, i + 1], [i + 2]] for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # ak.Array instances, axis=0
    arrays = [ak.Array([i, i + 1]) for i in range(5)]
    expected_list = ak.cartesian(arrays, axis=0)
    expected_tuple = ak.cartesian(tuple(arrays), axis=0)
    result = ak.cartesian((ak.Array([i, i + 1]) for i in range(5)), axis=0)
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # ak.Array instances, nested, default axis=1
    arrays = [ak.Array([[i, i + 1], [i + 2]]) for i in range(5)]
    expected_list = ak.cartesian(arrays)
    expected_tuple = ak.cartesian(tuple(arrays))
    result = ak.cartesian(ak.Array([[i, i + 1], [i + 2]]) for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)


def test_argcartesian():
    # Plain nested lists, default axis=1
    arrays = [[[i, i + 1], [i + 2]] for i in range(5)]
    expected_list = ak.argcartesian(arrays)
    expected_tuple = ak.argcartesian(tuple(arrays))
    result = ak.argcartesian([[i, i + 1], [i + 2]] for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # ak.Array instances, axis=0
    arrays = [ak.Array([i, i + 1]) for i in range(5)]
    expected_list = ak.argcartesian(arrays, axis=0)
    expected_tuple = ak.argcartesian(tuple(arrays), axis=0)
    result = ak.argcartesian((ak.Array([i, i + 1]) for i in range(5)), axis=0)
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)

    # ak.Array instances, nested, default axis=1
    arrays = [ak.Array([[i, i + 1], [i + 2]]) for i in range(5)]
    expected_list = ak.argcartesian(arrays)
    expected_tuple = ak.argcartesian(tuple(arrays))
    result = ak.argcartesian(ak.Array([[i, i + 1], [i + 2]]) for i in range(5))
    assert ak.array_equal(result, expected_list)
    assert ak.array_equal(result, expected_tuple)
