# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]

def test():
    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:2*3*5], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 25, 30], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  2,   3,   5,   7,  11],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109, 113]]]

    # assert awkward1.tolist(depth2.prod(axis=-1)) == [
    #      [  2 *   3 *   5 *   7 *  11,
    #        13 *  17 *  19 *  23 *  29,
    #        31 *  37 *  41 *  43 *  47],
    #      [ 53 *  59 *  61 *  67 *  71,
    #        73 *  79 *  83 *  89 *  97,
    #       101 * 103 * 107 * 109 * 113]]

    assert awkward1.tolist(depth2.prod(axis=-2)) == [
        [2*13*31, 3*17*37, 5*19*41, 7*23*43, 11*29*47],
        [53*73*101, 59*79*103, 61*83*107, 67*89*109, 71*97*113]]

    # content2 = awkward1.layout.NumpyArray(numpy.array(primes[:12], dtype=numpy.int64))
    # offsets3 = awkward1.layout.Index64(numpy.array([0, 4, 8, 12], dtype=numpy.int64))
    # depth1 = awkward1.layout.ListOffsetArray64(offsets3, content2)
