# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]

def test_reproduce_numpy():
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

    assert awkward1.tolist(depth2.prod(axis=-1)) == [
         [  2 *   3 *   5 *   7 *  11,
           13 *  17 *  19 *  23 *  29,
           31 *  37 *  41 *  43 *  47],
         [ 53 *  59 *  61 *  67 *  71,
           73 *  79 *  83 *  89 *  97,
          101 * 103 * 107 * 109 * 113]]
    assert awkward1.tolist(depth2.prod(axis=2)) == [
         [  2 *   3 *   5 *   7 *  11,
           13 *  17 *  19 *  23 *  29,
           31 *  37 *  41 *  43 *  47],
         [ 53 *  59 *  61 *  67 *  71,
           73 *  79 *  83 *  89 *  97,
          101 * 103 * 107 * 109 * 113]]

    assert awkward1.tolist(depth2.prod(axis=-2)) == [
        [2*13*31, 3*17*37, 5*19*41, 7*23*43, 11*29*47],
        [53*73*101, 59*79*103, 61*83*107, 67*89*109, 71*97*113]]
    assert awkward1.tolist(depth2.prod(axis=1)) == [
        [2*13*31, 3*17*37, 5*19*41, 7*23*43, 11*29*47],
        [53*73*101, 59*79*103, 61*83*107, 67*89*109, 71*97*113]]

    assert awkward1.tolist(depth2.prod(axis=-3)) == [
        [2*53, 3*59, 5*61, 7*67, 11*71],
        [13*73, 17*79, 19*83, 23*89, 29*97],
        [31*101, 37*103, 41*107, 43*109, 47*113]]
    assert awkward1.tolist(depth2.prod(axis=0)) == [
        [2*53, 3*59, 5*61, 7*67, 11*71],
        [13*73, 17*79, 19*83, 23*89, 29*97],
        [31*101, 37*103, 41*107, 43*109, 47*113]]

    content2 = awkward1.layout.NumpyArray(numpy.array(primes[:12], dtype=numpy.int64))
    offsets3 = awkward1.layout.Index64(numpy.array([0, 4, 8, 12], dtype=numpy.int64))
    depth1 = awkward1.layout.ListOffsetArray64(offsets3, content2)

    assert awkward1.tolist(depth1.prod(-1)) == [
        2*3*5*7,
        11*13*17*19,
        23*29*31*37]
    assert awkward1.tolist(depth1.prod(1)) == [
        2*3*5*7,
        11*13*17*19,
        23*29*31*37]

    assert awkward1.tolist(depth1.prod(-2)) == [
        2*11*23,
        3*13*29,
        5*17*31,
        7*19*37]
    assert awkward1.tolist(depth1.prod(0)) == [
        2*11*23,
        3*13*29,
        5*17*31,
        7*19*37]

def test_gaps():
    content1 = awkward1.layout.NumpyArray(numpy.array([123] + primes[:2*3*5], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 1, 6, 11, 16, 21, 26, 31], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([1, 4, 7], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  2,   3,   5,   7,  11],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109, 113]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 106,  177,  305,  469,  781],
        [ 949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:2*3*5 - 1], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 25, 29], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  2,   3,   5,   7,  11],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109,    ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 106,  177,  305,  469,  781],
        [ 949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687,   47]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:2*3*5 - 2], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 25, 28], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  2,   3,   5,   7,  11],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107,         ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 106,  177,  305,  469,  781],
        [ 949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387,   43,   47]]

    content1 = awkward1.layout.NumpyArray(numpy.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 101, 103, 107, 109], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 24, 28], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  2,   3,   5,   7,  11],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,    ],
         [101, 103, 107, 109     ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 106,  177,  305,  469, 781],
        [ 949, 1343, 1577, 2047,  29],
        [3131, 3811, 4387, 4687,  47]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[1:2*3*5], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 4, 9, 14, 19, 24, 29], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  3,   5,   7,  11     ],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109, 113]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 159,  295,  427,  737,   71],
        [ 949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[2:2*3*5], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 8, 13, 18, 23, 28], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  5,   7,  11          ],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109, 113]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 265,  413,  671,   67,   71],
        [ 949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311]]

    content1 = awkward1.layout.NumpyArray(numpy.array([3, 5, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 8, 13, 18, 23, 28], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  3,   5,   7,         ],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109, 113]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 159,  295,  427,   67,   71],
        [ 949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687, 5311]]

    content1 = awkward1.layout.NumpyArray(numpy.array([3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 4, 8, 13, 18, 23, 28], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  3,   5,   7,  11     ],
         [ 13,  17,  19,  23     ],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109, 113]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 159,  295,  427,  737,   71],
        [ 949, 1343, 1577, 2047,   97],
        [3131, 3811, 4387, 4687, 5311]]

    content1 = awkward1.layout.NumpyArray(numpy.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 5, 10, 14, 19, 24, 28], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  2,   3,   5,   7,  11],
         [ 13,  17,  19,  23,  29],
         [ 31,  37,  41,  43     ]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89,  97],
         [101, 103, 107, 109     ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 106,  177,  305,  469,  781],
        [ 949, 1343, 1577, 2047, 2813],
        [3131, 3811, 4387, 4687]]

    content1 = awkward1.layout.NumpyArray(numpy.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 101, 103, 107, 109, 113], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 5, 9, 14, 19, 23, 28], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[  2,   3,   5,   7,  11],
         [ 13,  17,  19,  23     ],
         [ 31,  37,  41,  43,  47]],
        [[ 53,  59,  61,  67,  71],
         [ 73,  79,  83,  89     ],
         [101, 103, 107, 109, 113]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 106,  177,  305,  469,  781],
        [ 949, 1343, 1577, 2047      ],
        [3131, 3811, 4387, 4687, 5311]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:9], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 4, 6, 6, 7, 9], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 2, 4, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[      2,   3, 5],
         [      7        ]],
        [[     11,  13   ],
         [               ]],
        [[     17        ],
         [     19,  23   ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [2*11*17, 3*13, 5],
        [7*19   , 23     ]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:9], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 4, 6, 7, 9], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 2, 3, 5], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[      2,   3, 5],
         [      7        ]],
        [[     11,  13   ]],
        [[     17        ],
         [     19,  23   ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [2*11*17, 3*13, 5],
        [7*19   , 23     ]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:10], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 5, 6, 8, 9, 10], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[ 2,  3, 5],
         [ 7, 11   ],
         [13       ]],
        [[17, 19   ],
         [23       ],
         [29       ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [ 34, 57, 5],
        [161, 11   ],
        [377       ]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:9], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 8, 9], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 4, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[ 2,  3, 5],
         [         ],
         [ 7, 11   ],
         [13       ]],
        [[17, 19   ],
         [23       ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [34, 57, 5],
        [23       ],
        [ 7, 11   ],
        [13       ]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:9], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 8, 9], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 4, 4, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[ 2,  3, 5],
         [         ],
         [ 7, 11   ],
         [13       ]],
        [],
        [[17, 19   ],
         [23       ]]]

    assert awkward1.tolist(depth2.prod(-3)) == [
        [34, 57, 5],
        [23       ],
        [ 7, 11   ],
        [13       ]]

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

    assert awkward1.tolist(depth2.prod(-1)) == [
        [  2 *   3 *   5 *   7 *  11,
          13 *  17 *  19 *  23 *  29,
          31 *  37 *  41 *  43 *  47],
        [ 53 *  59 *  61 *  67 *  71,
          73 *  79 *  83 *  89 *  97,
         101 * 103 * 107 * 109 * 113]]

    assert awkward1.tolist(depth2.prod(-2)) == [
        [2*13*31, 3*17*37, 5*19*41, 7*23*43, 11*29*47],
        [53*73*101, 59*79*103, 61*83*107, 67*89*109, 71*97*113]]

    content1 = awkward1.layout.NumpyArray(numpy.array(primes[:9], dtype=numpy.int64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 8, 9], dtype=numpy.int64))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 4, 4, 6], dtype=numpy.int64))
    depth2 = awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.ListOffsetArray64(offsets1, content1))
    assert awkward1.tolist(depth2) == [
        [[ 2,  3, 5],
         [         ],
         [ 7, 11   ],
         [13       ]],
        [],
        [[17, 19   ],
         [23       ]]]

    assert awkward1.tolist(depth2.prod(-1)) == [
        [2*3*5, 1, 7*11, 13],
        [],
        [17*19, 23]]

    assert awkward1.tolist(depth2.prod(-2)) == [
        [2*7*13, 3*11, 5],
        [],
        [17*23, 19]]

def test_complicated():
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    content1 = awkward1.layout.ListOffsetArray64(offsets1, awkward1.layout.NumpyArray(numpy.array(primes[:5], dtype=numpy.int64)))
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 8, 9], dtype=numpy.int64))
    offsets3 = awkward1.layout.Index64(numpy.array([0, 4, 4, 6], dtype=numpy.int64))
    content2 = awkward1.layout.ListOffsetArray64(offsets3, awkward1.layout.ListOffsetArray64(offsets2, awkward1.layout.NumpyArray(numpy.array(primes[:9], dtype=numpy.int64))))
    offsets4 = awkward1.layout.Index64(numpy.array([0, 1, 1, 3], dtype=numpy.int64))
    complicated = awkward1.layout.ListOffsetArray64(offsets4, awkward1.layout.RecordArray([content1, content2], ["x", "y"]))

    assert awkward1.tolist(complicated) == [[{"x": [2, 3, 5], "y": [[2, 3, 5], [], [7, 11], [13]]}], [], [{"x": [], "y": []}, {"x": [7, 11], "y": [[17, 19], [23]]}]]

    assert awkward1.tolist(complicated["x"]) == [
        [[2, 3, 5]],
        [],
        [[],
         [7, 11]]]
    assert awkward1.tolist(complicated["y"]) == [
        [[[ 2,  3, 5],
          [         ],
          [ 7, 11   ],
          [13       ]]],
        [             ],
        [[          ],
         [[17, 19   ],
          [23       ]]]]

    assert awkward1.tolist(complicated.prod(-1)) == [{"x": [30], "y": [[30, 1, 77, 13]]}, {"x": [], "y": []}, {"x": [1, 77], "y": [[], [323, 23]]}]
    assert awkward1.tolist(complicated["x"].prod(-1)) == [[30], [], [1, 77]]
    assert awkward1.tolist(complicated["y"].prod(-1)) == [[[30, 1, 77, 13]], [], [[], [323, 23]]]

    assert awkward1.tolist(complicated.prod(-2)) == [{"x": [2, 3, 5], "y": [[182, 33, 5]]}, {"x": [], "y": []}, {"x": [7, 11], "y": [[], [391, 19]]}]
    assert awkward1.tolist(complicated["x"].prod(-2)) == [[2, 3, 5], [], [7, 11]]
    assert awkward1.tolist(complicated["y"].prod(-2)) == [[[182, 33, 5]], [], [[], [391, 19]]]

    assert awkward1.tolist(complicated[0]) == [{"x": [2, 3, 5], "y": [[2, 3, 5], [], [7, 11], [13]]}]
    assert awkward1.tolist(complicated[0].prod(-1)) == {"x": [30], "y": [[30, 1, 77, 13]]}

def test_EmptyArray():
    offsets = awkward1.layout.Index64(numpy.array([0, 0, 0, 0], dtype=numpy.int64))
    array = awkward1.layout.ListOffsetArray64(offsets, awkward1.layout.EmptyArray())
    assert awkward1.tolist(array) == [[], [], []]

    assert awkward1.tolist(array.prod(-1)) == [1, 1, 1]

    offsets = awkward1.layout.Index64(numpy.array([0, 0, 0, 0], dtype=numpy.int64))
    array = awkward1.layout.ListOffsetArray64(offsets, awkward1.layout.NumpyArray(numpy.array([], dtype=numpy.int64)))
    assert awkward1.tolist(array) == [[], [], []]

    assert awkward1.tolist(array.prod(-1)) == [1, 1, 1]
