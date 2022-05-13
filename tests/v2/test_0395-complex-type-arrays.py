# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import numbers  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]


def test_count_complex():
    content2 = ak._v2.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak._v2.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak._v2.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [(1.1 + 0.1j), (2.2 + 0j), (3.3 + 0j)],
        [0j, (2.2 + 0j), 0j],
        [0j, (2.2 + 0j), 0j, (4.4 + 0j)],
    ]

    assert to_list(depth1.count(-1)) == [3, 3, 4]
    assert to_list(depth1.count(1)) == [3, 3, 4]

    assert to_list(depth1.count(-2)) == [3, 3, 3, 1]
    assert to_list(depth1.count(0)) == [3, 3, 3, 1]


def test_count_nonzero_complex():
    content2 = ak._v2.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak._v2.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak._v2.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [(1.1 + 0.1j), (2.2 + 0j), (3.3 + 0j)],
        [0j, (2.2 + 0j), 0j],
        [0j, (2.2 + 0j), 0j, (4.4 + 0j)],
    ]

    assert to_list(depth1.count_nonzero(-1)) == [3, 1, 2]
    assert to_list(depth1.count_nonzero(1)) == [3, 1, 2]

    assert to_list(depth1.count_nonzero(-2)) == [1, 3, 1, 1]
    assert to_list(depth1.count_nonzero(0)) == [1, 3, 1, 1]


def test_count_min_complex():
    content2 = ak._v2.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak._v2.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak._v2.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [(1.1 + 0.1j), (2.2 + 0j), (3.3 + 0j)],
        [0j, (2.2 + 0j), 0j],
        [0j, (2.2 + 0j), 0j, (4.4 + 0j)],
    ]

    assert to_list(depth1.min(-1)) == [(1.1 + 0.1j), 0j, 0j]
    assert to_list(depth1.min(1)) == [(1.1 + 0.1j), 0j, 0j]

    assert to_list(depth1.min(-2)) == [0j, (2.2 + 0j), 0j, (4.4 + 0j)]
    assert to_list(depth1.min(0)) == [0j, (2.2 + 0j), 0j, (4.4 + 0j)]

    content2 = ak._v2.contents.NumpyArray(
        np.array([True, True, True, False, True, False, False, True, False, True])
    )
    offsets3 = ak._v2.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak._v2.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [True, True, True],
        [False, True, False],
        [False, True, False, True],
    ]

    assert to_list(depth1.min(-1)) == [True, False, False]
    assert to_list(depth1.min(1)) == [True, False, False]

    assert to_list(depth1.min(-2)) == [False, True, False, True]
    assert to_list(depth1.min(0)) == [False, True, False, True]


def test_count_max_complex():
    content2 = ak._v2.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak._v2.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak._v2.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [(1.1 + 0.1j), (2.2 + 0j), (3.3 + 0j)],
        [0j, (2.2 + 0j), 0j],
        [0j, (2.2 + 0j), 0j, (4.4 + 0j)],
    ]

    assert to_list(depth1.max(-1)) == [3.3, 2.2, 4.4]
    assert to_list(depth1.max(1)) == [3.3, 2.2, 4.4]

    assert to_list(depth1.max(-2)) == [
        (1.1 + 0.1j),
        (2.2 + 0j),
        (3.3 + 0j),
        (4.4 + 0j),
    ]
    assert to_list(depth1.max(0)) == [
        (1.1 + 0.1j),
        (2.2 + 0j),
        (3.3 + 0j),
        (4.4 + 0j),
    ]

    content2 = ak._v2.contents.NumpyArray(
        np.array([False, True, True, False, True, False, False, False, False, False])
    )
    offsets3 = ak._v2.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak._v2.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [False, True, True],
        [False, True, False],
        [False, False, False, False],
    ]

    assert to_list(depth1.max(-1)) == [True, True, False]
    assert to_list(depth1.max(1)) == [True, True, False]

    assert to_list(depth1.max(-2)) == [False, True, True, False]
    assert to_list(depth1.max(0)) == [False, True, True, False]


def test_mask_complex():
    content = ak._v2.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 6, 6, 9], dtype=np.int64))
    array = ak._v2.contents.ListOffsetArray(offsets, content)

    assert to_list(array.min(axis=-1, mask=False)) == [
        (1.1 + 0.1j),
        (np.inf + 0j),
        (4.4 + 0j),
        (6.6 + 0j),
        (np.inf + 0j),
        (np.inf + 0j),
        (7.7 + 0j),
    ]
    assert to_list(array.min(axis=-1, mask=True)) == [
        (1.1 + 0.1j),
        None,
        (4.4 + 0j),
        (6.6 + 0j),
        None,
        None,
        (7.7 + 0j),
    ]


def test_keepdims_complex():
    nparray = np.array(primes[: 2 * 3 * 5], dtype=np.complex64).reshape(2, 3, 5)
    content1 = ak._v2.contents.NumpyArray(
        np.array(primes[: 2 * 3 * 5], dtype=np.complex64)
    )
    offsets1 = ak._v2.index.Index64(
        np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64)
    )
    offsets2 = ak._v2.index.Index64(np.array([0, 3, 6], dtype=np.int64))
    depth2 = ak._v2.contents.ListOffsetArray(
        offsets2, ak._v2.contents.ListOffsetArray(offsets1, content1)
    )
    assert to_list(depth2) == [
        [[2, 3, 5, 7, 11], [13, 17, 19, 23, 29], [31, 37, 41, 43, 47]],
        [[53, 59, 61, 67, 71], [73, 79, 83, 89, 97], [101, 103, 107, 109, 113]],
    ]

    assert to_list(depth2.prod(axis=-1, keepdims=False)) == to_list(
        nparray.prod(axis=-1, keepdims=False)
    )
    assert to_list(depth2.prod(axis=-2, keepdims=False)) == to_list(
        nparray.prod(axis=-2, keepdims=False)
    )
    assert to_list(depth2.prod(axis=-3, keepdims=False)) == to_list(
        nparray.prod(axis=-3, keepdims=False)
    )

    assert to_list(depth2.prod(axis=-1, keepdims=True)) == to_list(
        nparray.prod(axis=-1, keepdims=True)
    )
    assert to_list(depth2.prod(axis=-2, keepdims=True)) == to_list(
        nparray.prod(axis=-2, keepdims=True)
    )
    assert to_list(depth2.prod(axis=-3, keepdims=True)) == to_list(
        nparray.prod(axis=-3, keepdims=True)
    )


def test_astype_complex():
    content_float64 = ak._v2.contents.NumpyArray(
        np.array([0.25, 0.5, 3.5, 4.5, 5.5], dtype=np.float64)
    )
    assert content_float64.argmin() == 0
    assert content_float64.argmax() == 4

    array_float64 = ak._v2.contents.UnmaskedArray(content_float64)
    assert to_list(array_float64) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(ak._v2.operations.type(content_float64)) == "float64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(content_float64)))
        == "5 * float64"
    )
    assert str(ak._v2.operations.type(array_float64)) == "?float64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(array_float64)))
        == "5 * ?float64"
    )

    assert np.can_cast(np.float32, np.float64) is True
    assert np.can_cast(np.float64, np.float32, "unsafe") is True
    assert np.can_cast(np.float64, np.int8, "unsafe") is True
    assert np.can_cast(np.float64, np.complex64, "unsafe") is True
    assert np.can_cast(np.float64, np.complex128, "unsafe") is True
    assert np.can_cast(np.complex64, np.float64, "unsafe") is True
    assert np.can_cast(np.complex128, np.float64, "unsafe") is True

    content_complex64 = ak._v2.operations.values_astype(
        content_float64, "complex64", highlevel=False
    )
    array_complex64 = ak._v2.contents.UnmaskedArray(content_complex64)
    assert to_list(content_complex64) == [
        (0.25 + 0j),
        (0.5 + 0j),
        (3.5 + 0j),
        (4.5 + 0j),
        (5.5 + 0j),
    ]
    assert to_list(
        ak.nplike.of(array_complex64).asarray(ak._v2.highlevel.Array(array_complex64))
    ) == [
        (0.25 + 0.0j),
        (0.5 + 0.0j),
        (3.5 + 0.0j),
        (4.5 + 0.0j),
        (5.5 + 0.0j),
    ]
    assert str(ak._v2.operations.type(content_complex64)) == "complex64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(content_complex64)))
        == "5 * complex64"
    )
    assert str(ak._v2.operations.type(array_complex64)) == "?complex64"
    assert (
        str(ak._v2.operations.type(ak._v2.highlevel.Array(array_complex64)))
        == "5 * ?complex64"
    )
    content = ak._v2.contents.NumpyArray(
        np.array([1, (2.2 + 0.1j), 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    assert to_list(content) == [
        (1 + 0j),
        (2.2 + 0.1j),
        (3.3 + 0j),
        (4.4 + 0j),
        (5.5 + 0j),
        (6.6 + 0j),
        (7.7 + 0j),
        (8.8 + 0j),
        (9.9 + 0j),
    ]

    assert content_complex64.sum() == (14.25 + 0j)
    assert content_complex64.prod() == (10.828125 + 0j)
    assert content_complex64.min() == (0.25 + 0j)
    assert content_complex64.max() == (5.5 + 0j)
    assert content_complex64.argmin() == 0
    assert content_complex64.argmax() == 4
