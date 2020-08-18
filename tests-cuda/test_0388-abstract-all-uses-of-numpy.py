# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import awkward1
import awkward1.nplike


np = awkward1.nplike.NumpyMetadata.instance()
numpy = awkward1.nplike.Numpy.instance()
cupy = awkward1.nplike.Cupy.instance()


############################ array creation

def test_array():
    # data[, dtype=[, copy=]]
    cp_array = cupy.array([1.1, 2.2, 3.3, 4.4, 5.5])
    np_array = numpy.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert cp_array.tolist() == np_array.tolist()

    cp_array = cupy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=int)
    np_array = numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=int)
    assert cp_array.tolist() == np_array.tolist()

    cp_array1 = cupy.array([1.1, 2.2, 3.3, 4.4, 5.5])
    cp_array2 = cupy.array(cp_array1, copy=False)
    cp_array1[2] = 99.9
    assert cp_array2[2] == 99.9
    cp_array3 = cupy.array(cp_array1, copy=True)
    cp_array1[2] = 12.3
    assert cp_array3[2] == 99.9


def test_asarray():
    # array[, dtype=]
    cp_array1 = cupy.array([1.1, 2.2, 3.3, 4.4, 5.5])
    cp_array2 = cupy.asarray(cp_array1)
    cp_array1[2] = 99.9
    assert cp_array2[2] == 99.9

    cp_array3 = cupy.asarray(cp_array1, dtype=int)
    assert cp_array3.tolist() == [1, 2, 99, 4, 5]


def test_frombuffer():
    # array[, dtype=]
    np_array = numpy.frombuffer(b"one two three", dtype=np.uint8)
    cp_array = cupy.frombuffer(b"one two three", dtype=np.uint8)
    assert cp_array.tolist() == np_array.tolist()


def test_zeros():
    # shape/len[, dtype=]
    assert cupy.zeros(10).tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert cupy.zeros(10, dtype=int).tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert cupy.zeros((2, 3)).tolist() == [[0, 0, 0], [0, 0, 0]]
    assert cupy.zeros((2, 3), dtype=int).tolist() == [[0, 0, 0], [0, 0, 0]]


def test_ones():
    # shape/len[, dtype=]
    assert cupy.ones(10).tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert cupy.ones(10, dtype=int).tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert cupy.ones((2, 3)).tolist() == [[1, 1, 1], [1, 1, 1]]
    assert cupy.ones((2, 3), dtype=int).tolist() == [[1, 1, 1], [1, 1, 1]]


def test_empty():
    # shape/len[, dtype=]
    assert cupy.empty(10).shape == (10,)
    assert cupy.empty(10, dtype=int).shape == (10,)
    assert cupy.empty((2, 3)).shape == (2, 3)
    assert cupy.empty((2, 3), dtype=int).shape == (2, 3)


def test_full():
    # shape/len, value[, dtype=]
    assert cupy.full(10, 5).tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    assert cupy.full(10, 5, dtype=int).tolist() == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    assert cupy.full((2, 3), 5).tolist() == [[5, 5, 5], [5, 5, 5]]
    assert cupy.full((2, 3), 5, dtype=int).tolist() == [[5, 5, 5], [5, 5, 5]]


def test_arange():
    # stop[, dtype=]
    # start, stop[, dtype=]
    # start, stop, step[, dtype=]
    assert cupy.arange(10).tolist() == numpy.arange(10).tolist()
    assert cupy.arange(10, 15).tolist() == numpy.arange(10, 15).tolist()
    assert cupy.arange(10, 30, 5).tolist() == numpy.arange(10, 30, 5).tolist()
    assert cupy.arange(10, dtype=float).dtype.kind == "f"
    assert cupy.arange(10, 15, dtype=float).dtype.kind == "f"
    assert cupy.arange(10, 30, 5, dtype=float).dtype.kind == "f"


def test_meshgrid():
    # *arrays, indexing="ij"
    np_one = numpy.array([2, 3, 5, 7, 9])
    np_two = numpy.array([11, 13, 17, 19])
    np_three = numpy.array([21, 23, 29])
    np_a, np_b = numpy.meshgrid(np_one, np_two, indexing="ij")
    np_x, np_y, np_z = numpy.meshgrid(np_one, np_two, np_three, indexing="ij")

    cp_one = cupy.array([2, 3, 5, 7, 9])
    cp_two = cupy.array([11, 13, 17, 19])
    cp_three = cupy.array([21, 23, 29])
    cp_a, cp_b = cupy.meshgrid(cp_one, cp_two, indexing="ij")
    cp_x, cp_y, cp_z = cupy.meshgrid(cp_one, cp_two, cp_three, indexing="ij")

    assert np_a.tolist() == cp_a.tolist()
    assert np_b.tolist() == cp_b.tolist()
    assert np_x.tolist() == cp_x.tolist()
    assert np_y.tolist() == cp_y.tolist()
    assert np_z.tolist() == cp_z.tolist()


############################ testing

def test_array_equal():
    # array1, array2
    assert numpy.array_equal(numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]), numpy.array([1, 2, 3, 4, 5]))
    assert cupy.array_equal(cupy.array([1.0, 2.0, 3.0, 4.0, 5.0]), cupy.array([1, 2, 3, 4, 5]))


def test_size():
    # array
    assert numpy.size(numpy.arange(2*3*5).reshape(2, 3, 5)) == cupy.size(cupy.arange(2*3*5).reshape(2, 3, 5))


def test_searchsorted():
    # haystack, needle, side="right"
    np_haystack = numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    np_needle = numpy.array([2.1, 5.3, 5.5, 5.6, 3.8, 7.0, 2.1, 3.3, -2.0, 15.0])
    assert numpy.searchsorted(np_haystack, np_needle).tolist() == [2, 5, 5, 6, 4, 7, 2, 3, 0, 10]

    cp_haystack = cupy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    cp_needle = cupy.array([2.1, 5.3, 5.5, 5.6, 3.8, 7.0, 2.1, 3.3, -2.0, 15.0])
    assert cupy.searchsorted(cp_haystack, cp_needle).tolist() == [2, 5, 5, 6, 4, 7, 2, 3, 0, 10]


############################ manipulation

def test_cumsum():
    # arrays[, out=]
    np_prefixsum = numpy.empty(5)
    assert numpy.cumsum([3, 5, 5, 0, 2], out=np_prefixsum).tolist() == [3, 8, 13, 13, 15]
    assert np_prefixsum.tolist() == [3, 8, 13, 13, 15]

    cp_prefixsum = cupy.empty(5)
    assert cupy.cumsum([3, 5, 5, 0, 2], out=cp_prefixsum).tolist() == [3, 8, 13, 13, 15]
    assert cp_prefixsum.tolist() == [3, 8, 13, 13, 15]


def test_nonzero():
    # array
    out, = numpy.nonzero(numpy.array([3.3, 2.2, 0.0, 5.5, 0.0, 0.0, 2.2]))
    assert out.tolist() == [0, 1, 3, 6]

    out, = cupy.nonzero(cupy.array([3.3, 2.2, 0.0, 5.5, 0.0, 0.0, 2.2]))
    assert out.tolist() == [0, 1, 3, 6]


def test_unique():
    # array
    assert numpy.unique(numpy.array([3.3, 2.2, 5.5, 2.2, 6.6, 2.2, 3.3, 0.0, 1.1, 5.5])).tolist() == [0.0, 1.1, 2.2, 3.3, 5.5, 6.6]

    assert cupy.unique(cupy.array([3.3, 2.2, 5.5, 2.2, 6.6, 2.2, 3.3, 0.0, 1.1, 5.5])).tolist() == [0.0, 1.1, 2.2, 3.3, 5.5, 6.6]


def test_concatenate():
    # arrays
    np_one = numpy.array([2, 3, 5, 7, 9])
    np_two = numpy.array([11.1, 13.3, 17.7, 19.9])
    np_three = numpy.array([21, 23, 29])
    assert numpy.concatenate([np_one, np_two, np_three]).tolist() == [2, 3, 5, 7, 9, 11.1, 13.3, 17.7, 19.9, 21, 23, 29]

    cp_one = cupy.array([2, 3, 5, 7, 9])
    cp_two = cupy.array([11.1, 13.3, 17.7, 19.9])
    cp_three = cupy.array([21, 23, 29])
    assert cupy.concatenate([cp_one, cp_two, cp_three]).tolist() == [2, 3, 5, 7, 9, 11.1, 13.3, 17.7, 19.9, 21, 23, 29]


def test_repeat():
    # array, int
    # array1, array2
    np_one = numpy.array([2.2, 3.3, 5.5, 7.7, 9.9])
    np_two = numpy.array([2, 0, 1, 4, 3])
    assert numpy.repeat(np_one, 3).tolist() == [2.2, 2.2, 2.2, 3.3, 3.3, 3.3, 5.5, 5.5, 5.5, 7.7, 7.7, 7.7, 9.9, 9.9, 9.9]
    assert numpy.repeat(np_one, np_two).tolist() == [2.2, 2.2, 5.5, 7.7, 7.7, 7.7, 7.7, 9.9, 9.9, 9.9]

    cp_one = cupy.array([2.2, 3.3, 5.5, 7.7, 9.9])
    cp_two = cupy.array([2, 0, 1, 4, 3])
    assert cupy.repeat(cp_one, 3).tolist() == [2.2, 2.2, 2.2, 3.3, 3.3, 3.3, 5.5, 5.5, 5.5, 7.7, 7.7, 7.7, 9.9, 9.9, 9.9]
    assert cupy.repeat(cp_one, cp_two).tolist() == [2.2, 2.2, 5.5, 7.7, 7.7, 7.7, 7.7, 9.9, 9.9, 9.9]


def test_stack():
    # arrays
    assert numpy.stack([numpy.array([[0, 1, 2], [3, 4, 5]]), numpy.array([[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]])]).tolist() == [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]]
    assert cupy.stack([cupy.array([[0, 1, 2], [3, 4, 5]]), cupy.array([[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]])]).tolist() == [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]]


def test_vstack():
    # arrays
    assert numpy.vstack([numpy.array([[0, 1, 2], [3, 4, 5]]), numpy.array([[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]])]).tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]
    assert cupy.vstack([cupy.array([[0, 1, 2], [3, 4, 5]]), cupy.array([[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]])]).tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]


def test_packbits():
    # array
    assert numpy.packbits(numpy.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0])).tolist() == [29, 8]
    assert cupy.packbits(cupy.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0])).tolist() == [29, 8]


def test_unpackbits():
    # array
    assert numpy.unpackbits(numpy.array([29, 8], dtype=np.uint8)).tolist() == [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    assert cupy.unpackbits(cupy.array([29, 8], dtype=np.uint8)).tolist() == [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]


def test_atleast_1d():
    # *arrays
    assert numpy.atleast_1d(numpy.array([1.1, 2.2, 3.3])).tolist() == [1.1, 2.2, 3.3]
    assert cupy.atleast_1d(cupy.array([1.1, 2.2, 3.3])).tolist() == [1.1, 2.2, 3.3]


def test_broadcast_to():
    # array, shape
    assert numpy.broadcast_to(numpy.array([[1], [2], [3], [4], [5]]), (5, 2)).tolist() == [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    assert numpy.broadcast_to(numpy.array([1, 2, 3, 4, 5]), (2, 5)).tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    assert cupy.broadcast_to(cupy.array([[1], [2], [3], [4], [5]]), (5, 2)).tolist() == [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    assert cupy.broadcast_to(cupy.array([1, 2, 3, 4, 5]), (2, 5)).tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]


############################ ufuncs

def test_sqrt():
    # array
    assert numpy.sqrt(numpy.array([1, 4, 9, 16, 25])).tolist() == cupy.sqrt(cupy.array([1, 4, 9, 16, 25])).tolist()
    assert numpy.sqrt(numpy.array([1.1, 4.4, 9.9, 16.16, 25.25])).tolist() == cupy.sqrt(cupy.array([1.1, 4.4, 9.9, 16.16, 25.25])).tolist()


def test_exp():
    # array
    assert numpy.exp(numpy.array([1, 2, 3, 4, 5])).tolist() == cupy.exp(cupy.array([1, 2, 3, 4, 5])).tolist()
    assert numpy.exp(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5])).tolist() == cupy.exp(cupy.array([1.1, 2.2, 3.3, 4.4, 5.5])).tolist()


def test_true_divide():
    # array1, array2
    assert numpy.true_divide(numpy.array([1.1, 2.2, 3.3]), numpy.array([1.0, 10.0, 7.0])).tolist() == cupy.true_divide(cupy.array([1.1, 2.2, 3.3]), cupy.array([1.0, 10.0, 7.0])).tolist()


def test_bitwise_or():
    # array1, array2[, out=output]
    np_array1 = numpy.array([0, 0, 1, 1, 4, 7, 0, 1], dtype=np.uint8)
    np_array2 = numpy.array([0, 1, 0, 1, 4, 4, 0, 5], dtype=np.uint8)
    np_arrayout = numpy.zeros(8, dtype=np.uint8)

    cp_array1 = cupy.array([0, 0, 1, 1, 4, 7, 0, 1], dtype=np.uint8)
    cp_array2 = cupy.array([0, 1, 0, 1, 4, 4, 0, 5], dtype=np.uint8)
    cp_arrayout = cupy.ones(8, dtype=np.uint8)

    assert numpy.bitwise_or(np_array1, np_array2).tolist() == cupy.bitwise_or(cp_array1, cp_array2).tolist()
    numpy.bitwise_or(np_array1, np_array2, np_arrayout)
    cupy.bitwise_or(cp_array1, cp_array2, cp_arrayout)
    assert np_arrayout.tolist() == cp_arrayout.tolist()


def test_logical_and():
    # array1, array2
    assert numpy.logical_and(numpy.array([False, False, True, True]), numpy.array([False, True, False, True])).tolist() == cupy.logical_and(cupy.array([False, False, True, True]), cupy.array([False, True, False, True])).tolist()


def test_equal():
    # array1, array2
    assert numpy.equal(numpy.array([3.3, 2.2, 5.5, 2.2, 1.1]), numpy.array([4.4, 2.2, 5.5, 1.1, 5.5])).tolist() == cupy.equal(cupy.array([3.3, 2.2, 5.5, 2.2, 1.1]), cupy.array([4.4, 2.2, 5.5, 1.1, 5.5])).tolist()


def test_ceil():
    # array
    assert numpy.ceil(numpy.array([3.3, 2.0, 1.9, 2.1, 5.5, 1.9, 2.0, 2.1])).tolist() == cupy.ceil(cupy.array([3.3, 2.0, 1.9, 2.1, 5.5, 1.9, 2.0, 2.1])).tolist()
    assert numpy.ceil(numpy.array([3.3, 2.0, 1.9, 2.1, 5.5, 1.9, 2.0, 2.1])).dtype.kind == "f"
    assert cupy.ceil(cupy.array([3.3, 2.0, 1.9, 2.1, 5.5, 1.9, 2.0, 2.1])).dtype.kind == "f"


############################ reducers

def test_all():
    # array
    assert numpy.all(numpy.array([])) == cupy.all(cupy.array([]))
    assert numpy.all(numpy.array([False, False])) == cupy.all(cupy.array([False, False]))
    assert numpy.all(numpy.array([False, True])) == cupy.all(cupy.array([False, True]))
    assert numpy.all(numpy.array([True, True])) == cupy.all(cupy.array([True, True]))


def test_any():
    # array
    assert numpy.any(numpy.array([])) == cupy.any(cupy.array([]))
    assert numpy.any(numpy.array([False, False])) == cupy.any(cupy.array([False, False]))
    assert numpy.any(numpy.array([False, True])) == cupy.any(cupy.array([False, True]))
    assert numpy.any(numpy.array([True, True])) == cupy.any(cupy.array([True, True]))


def test_count_nonzero():
    # array
    assert numpy.count_nonzero(numpy.array([])) == cupy.count_nonzero(cupy.array([]))
    assert numpy.count_nonzero(numpy.array([2, 0, 0, 1])) == cupy.count_nonzero(cupy.array([2, 0, 0, 1]))


def test_sum():
    # array
    assert numpy.sum(numpy.array([2, 5, 0, 4])) == cupy.sum(cupy.array([2, 5, 0, 4]))


def test_prod():
    # array
    assert numpy.prod(numpy.array([2, 3, 5, 7])) == cupy.prod(cupy.array([2, 3, 5, 7]))


def test_min():
    # array
    assert numpy.min(numpy.array([2, 5, 1, 4])) == cupy.min(cupy.array([2, 5, 1, 4]))


def test_max():
    # array
    assert numpy.max(numpy.array([2, 5, 1, 4])) == cupy.max(cupy.array([2, 5, 1, 4]))


def test_argmin():
    # array[, axis=]
    assert numpy.argmin(numpy.array([2, 5, 1, 4]), axis=None) == cupy.argmin(cupy.array([2, 5, 1, 4]), axis=None)
    assert numpy.argmin(numpy.array([[2, 5], [1, 4]]), axis=None) == cupy.argmin(cupy.array([[2, 5], [1, 4]]), axis=None)


def test_argmax():
    # array[, axis=]
    assert numpy.argmax(numpy.array([2, 5, 1, 4]), axis=None) == cupy.argmax(cupy.array([2, 5, 1, 4]), axis=None)
    assert numpy.argmax(numpy.array([[2, 5], [1, 4]]), axis=None) == cupy.argmax(cupy.array([[2, 5], [1, 4]]), axis=None)
