# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_mixing_lists_and_none():
    def add(a, b):
        outer = []
        for ai, bi in zip(a, b):
            if ai is None or bi is None:
                outer.append(None)
            else:
                inner = []
                for aj, bj in zip(ai, bi):
                    if aj is None or bj is None:
                        inner.append(None)
                    else:
                        inner.append(aj + bj)
                outer.append(inner)
        return outer

    a00 = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]], check_valid=True
    )
    a01 = ak.highlevel.Array(
        [[1.1, None, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]], check_valid=True
    )
    a02 = ak.highlevel.Array(
        [[1.1, None, 3.3], [], [4.4, 5.5], [None, 7.7, 8.8, 9.9]], check_valid=True
    )
    a10 = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], None, [6.6, 7.7, 8.8, 9.9]], check_valid=True
    )
    a11 = ak.highlevel.Array(
        [[1.1, None, 3.3], [], None, [6.6, 7.7, 8.8, 9.9]], check_valid=True
    )
    a12 = ak.highlevel.Array(
        [[1.1, None, 3.3], [], None, [None, 7.7, 8.8, 9.9]], check_valid=True
    )
    a20 = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], None, None, [6.6, 7.7, 8.8, 9.9]], check_valid=True
    )
    a21 = ak.highlevel.Array(
        [[1.1, None, 3.3], None, None, [6.6, 7.7, 8.8, 9.9]], check_valid=True
    )
    a22 = ak.highlevel.Array(
        [[1.1, None, 3.3], None, None, [None, 7.7, 8.8, 9.9]], check_valid=True
    )

    b00 = ak.highlevel.Array(
        [[100, 200, 300], [], [400, 500], [600, 700, 800, 900]], check_valid=True
    )
    b01 = ak.highlevel.Array(
        [[100, None, 300], [], [400, 500], [600, 700, 800, 900]], check_valid=True
    )
    b02 = ak.highlevel.Array(
        [[100, None, 300], [], [400, 500], [None, 700, 800, 900]], check_valid=True
    )
    b10 = ak.highlevel.Array(
        [[100, 200, 300], [], None, [600, 700, 800, 900]], check_valid=True
    )
    b11 = ak.highlevel.Array(
        [[100, None, 300], [], None, [600, 700, 800, 900]], check_valid=True
    )
    b12 = ak.highlevel.Array(
        [[100, None, 300], [], None, [None, 700, 800, 900]], check_valid=True
    )
    b20 = ak.highlevel.Array(
        [[100, 200, 300], None, None, [600, 700, 800, 900]], check_valid=True
    )
    b21 = ak.highlevel.Array(
        [[100, None, 300], None, None, [600, 700, 800, 900]], check_valid=True
    )
    b22 = ak.highlevel.Array(
        [[100, None, 300], None, None, [None, 700, 800, 900]], check_valid=True
    )

    for a in (a00, a01, a02, a10, a11, a12, a20, a21, a22):
        for b in (b00, b01, b02, b10, b11, b12, b20, b21, b22):
            assert to_list(a + b) == add(a, b)


def test_explicit_broadcasting():
    nparray = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    lsarray = ak.highlevel.Array(nparray.tolist(), check_valid=True)
    rgarray = ak.highlevel.Array(nparray, check_valid=True)

    # explicit left-broadcasting
    assert to_list(rgarray + np.array([[[100]], [[200]]])) == to_list(
        nparray + np.array([[[100]], [[200]]])
    )
    assert to_list(lsarray + np.array([[[100]], [[200]]])) == to_list(
        nparray + np.array([[[100]], [[200]]])
    )
    assert to_list(np.array([[[100]], [[200]]]) + rgarray) == to_list(
        np.array([[[100]], [[200]]]) + nparray
    )
    assert to_list(np.array([[[100]], [[200]]]) + lsarray) == to_list(
        np.array([[[100]], [[200]]]) + nparray
    )

    # explicit right-broadcasting
    assert to_list(rgarray + np.array([[[100, 200, 300, 400, 500]]])) == to_list(
        nparray + np.array([[[100, 200, 300, 400, 500]]])
    )
    assert to_list(lsarray + np.array([[[100, 200, 300, 400, 500]]])) == to_list(
        nparray + np.array([[[100, 200, 300, 400, 500]]])
    )
    assert to_list(np.array([[[100, 200, 300, 400, 500]]]) + rgarray) == to_list(
        np.array([[[100, 200, 300, 400, 500]]]) + nparray
    )
    assert to_list(np.array([[[100, 200, 300, 400, 500]]]) + lsarray) == to_list(
        np.array([[[100, 200, 300, 400, 500]]]) + nparray
    )


def test_implicit_broadcasting():
    nparray = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    lsarray = ak.highlevel.Array(nparray.tolist(), check_valid=True)
    rgarray = ak.highlevel.Array(nparray, check_valid=True)

    assert to_list(rgarray + np.array([100, 200, 300, 400, 500])) == to_list(
        nparray + np.array([100, 200, 300, 400, 500])
    )
    assert to_list(np.array([100, 200, 300, 400, 500]) + rgarray) == to_list(
        np.array([100, 200, 300, 400, 500]) + nparray
    )

    assert to_list(lsarray + np.array([100, 200])) == to_list(
        nparray + np.array([[[100]], [[200]]])
    )
    assert to_list(np.array([100, 200]) + lsarray) == to_list(
        np.array([[[100]], [[200]]]) + nparray
    )


def test_tonumpy():
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)
        ),
        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
    )
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array(np.array([1.1, 2.2, 3.3, 4.4, 5.5]), check_valid=True)
        ),
        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
    )
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], check_valid=True)
        ),
        np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]),
    )
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array(
                np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]), check_valid=True
            )
        ),
        np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]),
    )
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array(["one", "two", "three"], check_valid=True)
        ),
        np.array(["one", "two", "three"]),
    )
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array([b"one", b"two", b"three"], check_valid=True)
        ),
        np.array([b"one", b"two", b"three"]),
    )
    assert np.array_equal(
        ak.operations.to_numpy(ak.highlevel.Array([], check_valid=True)),
        np.array([]),
    )

    content0 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    )
    content1 = ak.from_iter(["1", "2", "3"], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 1, 0, 0, 0, 1, 0], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 3, 2, 4], dtype=np.int64))
    array = ak.highlevel.Array(
        ak.contents.UnionArray(tags, index, [content0, content1]), check_valid=True
    )
    assert ak.operations.to_numpy(array).tolist() == [
        "1.1",
        "1",
        "2",
        "2.2",
        "3.3",
        "4.4",
        "3",
        "5.5",
    ]

    assert ak.operations.to_numpy(
        ak.highlevel.Array([1.1, 2.2, None, None, 3.3], check_valid=True)
    ).tolist() == [1.1, 2.2, None, None, 3.3]
    assert ak.operations.to_numpy(
        ak.highlevel.Array([[1.1, 2.2], [None, None], [3.3, 4.4]], check_valid=True)
    ).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]
    assert ak.operations.to_numpy(
        ak.highlevel.Array([[1.1, 2.2], None, [3.3, 4.4]], check_valid=True)
    ).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]

    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array(
                [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                check_valid=True,
            )
        ),
        np.array(
            [(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("x", np.int64), ("y", np.float64)]
        ),
    )
    assert np.array_equal(
        ak.operations.to_numpy(
            ak.highlevel.Array([(1, 1.1), (2, 2.2), (3, 3.3)], check_valid=True)
        ),
        np.array(
            [(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("0", np.int64), ("1", np.float64)]
        ),
    )


def test_numpy_array():
    assert np.array_equal(
        np.asarray(ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)),
        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
    )
    assert np.array_equal(
        np.asarray(
            ak.highlevel.Array(np.array([1.1, 2.2, 3.3, 4.4, 5.5]), check_valid=True)
        ),
        np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
    )
    assert np.array_equal(
        np.asarray(
            ak.highlevel.Array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], check_valid=True)
        ),
        np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]),
    )
    assert np.array_equal(
        np.asarray(
            ak.highlevel.Array(
                np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]), check_valid=True
            )
        ),
        np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]),
    )
    assert np.array_equal(
        np.asarray(ak.highlevel.Array(["one", "two", "three"], check_valid=True)),
        np.array(["one", "two", "three"]),
    )
    assert np.array_equal(
        np.asarray(ak.highlevel.Array([b"one", b"two", b"three"], check_valid=True)),
        np.array([b"one", b"two", b"three"]),
    )
    assert np.array_equal(
        np.asarray(ak.highlevel.Array([], check_valid=True)), np.array([])
    )

    content0 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    )
    content1 = ak.from_iter(["1", "2", "3"], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 1, 0, 0, 0, 1, 0], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 3, 2, 4], dtype=np.int64))
    array = ak.highlevel.Array(
        ak.contents.UnionArray(tags, index, [content0, content1]), check_valid=True
    )
    assert np.asarray(array).tolist() == [
        "1.1",
        "1",
        "2",
        "2.2",
        "3.3",
        "4.4",
        "3",
        "5.5",
    ]

    assert ak.operations.to_numpy(
        ak.highlevel.Array([1.1, 2.2, None, None, 3.3], check_valid=True)
    ).tolist() == [1.1, 2.2, None, None, 3.3]
    assert ak.operations.to_numpy(
        ak.highlevel.Array([[1.1, 2.2], [None, None], [3.3, 4.4]], check_valid=True)
    ).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]
    assert ak.operations.to_numpy(
        ak.highlevel.Array([[1.1, 2.2], None, [3.3, 4.4]], check_valid=True)
    ).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]


def test_numpy_array_FIXME():
    assert np.array_equal(
        np.asarray(
            ak.highlevel.Array(
                [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                check_valid=True,
            )
        ),
        np.array(
            [(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("x", np.int64), ("y", np.float64)]
        ),
    )
    assert np.array_equal(
        np.asarray(
            ak.highlevel.Array([(1, 1.1), (2, 2.2), (3, 3.3)], check_valid=True)
        ),
        np.array(
            [(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("0", np.int64), ("1", np.float64)]
        ),
    )


def test_where_FIXME():
    condition = ak.highlevel.Array(
        [False, False, False, False, False, True, False, True, False, True],
        check_valid=True,
    )

    assert isinstance(ak.operations.where(condition)[0], ak.highlevel.Array)


def test_where():
    one = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    )
    two = ak.highlevel.Array(
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900], check_valid=True
    )
    condition = ak.highlevel.Array(
        [False, False, False, False, False, True, False, True, False, True],
        check_valid=True,
    )

    assert isinstance(ak.operations.where(condition)[0], ak.highlevel.Array)

    assert to_list(ak.operations.where(condition)[0]) == [5, 7, 9]
    assert to_list(ak.operations.where(condition, one, two)) == to_list(
        np.where(np.asarray(condition), np.asarray(one), np.asarray(two))
    )


def test_string_equal():
    one = ak.highlevel.Array(["one", "two", "three"], check_valid=True)
    two = ak.highlevel.Array(["ONE", "two", "four"], check_valid=True)
    assert to_list(one == two) == [False, True, False]
