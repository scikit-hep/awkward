# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import pytest
import numpy

import awkward1

def test_listarray():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops1  = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 9]))
    array1  = awkward1.layout.ListArray64(starts1, stops1, content)
    starts2 = awkward1.layout.Index64(numpy.array([0, 2, 3, 3]))
    stops2  = awkward1.layout.Index64(numpy.array([2, 3, 3, 5]))
    array2  = awkward1.layout.ListArray64(starts2, stops2, array1)

    # assert awkward1.tolist(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    # assert awkward1.tolist(array1[2]) == [4.4, 5.5]
    # assert awkward1.tolist(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    # assert awkward1.tolist(array2) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    # assert awkward1.tolist(array2[1]) == [[4.4, 5.5]]
    # assert awkward1.tolist(array2[1:-1]) == [[[4.4, 5.5]], []]
    #
    # assert awkward1.tolist(array1[numpy.array([2, 0, 0, 1, -1])]) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]
    # assert awkward1.tolist(array1[numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0])]) == [5.5, 2.2, 1.1, 7.7]

    content_deep = awkward1.layout.NumpyArray(numpy.array([[0, 0], [1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]))
    starts1_deep = awkward1.layout.Index64(numpy.array([0, 3, 6]))
    stops1_deep  = awkward1.layout.Index64(numpy.array([3, 6, 9]))
    array1_deep  = awkward1.layout.ListArray64(starts1_deep, stops1_deep, content_deep)
    # assert awkward1.tolist(array1_deep) == [[[0, 0], [1, 10], [2, 20]], [], [[3, 30], [4, 40]], [[5, 50]], [[6, 60], [7, 70], [8, 80]]]
    # print(awkward1.tolist(array1_deep[numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0]), numpy.array([0, 1, 0, 1])]))
    # [[4, 40], [1, 10], [0, 0], [6, 60]]
    # [4, 10, 0, 60]

    assert awkward1.tolist(array1_deep) == [[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]]
    s = (numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0]), numpy.array([0, 1, 0, 1]))
    # s = (slice(1, None), slice(None, 1))
    print(numpy.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist())
    print(awkward1.tolist(array1_deep[s]))

    assert numpy.array([[[0, 0], [1, 10], [2, 20]], [[3, 30], [4, 40], [5, 50]], [[6, 60], [7, 70], [8, 80]]])[s].tolist() == awkward1.tolist(array1_deep[s])

    # raise Exception
