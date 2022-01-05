# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import itertools

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

to_list = ak._v2.operations.convert.to_list

content = ak.layout.NumpyArray(
    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
)
offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
regulararray = ak.layout.RegularArray(listoffsetarray, 2, zeros_length=0)
starts = ak.layout.Index64(np.array([0, 1]))
stops = ak.layout.Index64(np.array([2, 3]))
listarray = ak.layout.ListArray64(starts, stops, regulararray)

regulararray = v1_to_v2(regulararray)
listarray = v1_to_v2(listarray)


def test_iteration():
    assert to_list(regulararray) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]


def test_getitem_at():
    assert to_list(regulararray[0]) == [[0.0, 1.1, 2.2], []]
    assert to_list(regulararray[1]) == [[3.3, 4.4], [5.5]]
    assert to_list(regulararray[2]) == [[6.6, 7.7, 8.8, 9.9], []]
    assert regulararray.typetracer[2].form == regulararray[2].form


def test_getitem_range():
    assert to_list(regulararray[1:]) == [
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert regulararray.typetracer[1:].form == regulararray[1:].form
    assert to_list(regulararray[:-1]) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]
    assert regulararray.typetracer[:-1].form == regulararray[:-1].form


def test_getitem():
    assert to_list(regulararray[(0,)]) == [[0.0, 1.1, 2.2], []]
    assert to_list(regulararray[(1,)]) == [[3.3, 4.4], [5.5]]
    assert to_list(regulararray[(2,)]) == [[6.6, 7.7, 8.8, 9.9], []]
    assert regulararray.typetracer[(2,)].form == regulararray[(2,)].form
    assert to_list(regulararray[(slice(1, None, None),)]) == [
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer[(slice(1, None, None),)].form
        == regulararray[(slice(1, None, None),)].form
    )
    assert to_list(regulararray[(slice(None, -1, None),)]) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
    ]
    assert (
        regulararray.typetracer[(slice(None, -1, None),)].form
        == regulararray[(slice(None, -1, None),)].form
    )


def test_getitem_deeper():
    assert to_list(listarray) == [
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]],
        [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]],
    ]

    assert to_list(listarray[0, 0, 0]) == [0.0, 1.1, 2.2]
    assert to_list(listarray[0, 0, 1]) == []
    assert to_list(listarray[0, 1, 0]) == [3.3, 4.4]
    assert to_list(listarray[0, 1, 1]) == [5.5]
    assert to_list(listarray[1, 0, 0]) == [3.3, 4.4]
    assert to_list(listarray[1, 0, 1]) == [5.5]
    assert to_list(listarray[1, 1, 0]) == [6.6, 7.7, 8.8, 9.9]
    assert to_list(listarray[1, 1, 1]) == []
    assert listarray.typetracer[1, 1, 1].form == listarray[1, 1, 1].form

    assert to_list(listarray[0, 0, 0:]) == [[0.0, 1.1, 2.2], []]
    assert to_list(listarray[0, 0, 1:]) == [[]]
    assert to_list(listarray[0, 1, 0:]) == [[3.3, 4.4], [5.5]]
    assert to_list(listarray[0, 1, 1:]) == [[5.5]]
    assert to_list(listarray[1, 0, 0:]) == [[3.3, 4.4], [5.5]]
    assert to_list(listarray[1, 0, 1:]) == [[5.5]]
    assert to_list(listarray[1, 1, 0:]) == [[6.6, 7.7, 8.8, 9.9], []]
    assert to_list(listarray[1, 1, 1:]) == [[]]
    assert listarray.typetracer[1, 1, 1:].form == listarray[1, 1, 1:].form

    assert to_list(listarray[[1], 0, 0:]) == [[[3.3, 4.4], [5.5]]]
    assert listarray.typetracer[[1], 0, 0:].form == listarray[[1], 0, 0:].form
    assert to_list(listarray[[1, 0], 0, 0:]) == [
        [[3.3, 4.4], [5.5]],
        [[0.0, 1.1, 2.2], []],
    ]
    assert listarray.typetracer[[1, 0], 0, 0:].form == listarray[[1, 0], 0, 0:].form

    assert to_list(listarray[:, :, [0, 1]]) == [
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]],
        [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]],
    ]
    assert to_list(listarray[:, :, [1, 0]]) == [
        [[[], [0.0, 1.1, 2.2]], [[5.5], [3.3, 4.4]]],
        [[[5.5], [3.3, 4.4]], [[], [6.6, 7.7, 8.8, 9.9]]],
    ]
    assert to_list(listarray[:, :, [1, 0, 1]]) == [
        [[[], [0.0, 1.1, 2.2], []], [[5.5], [3.3, 4.4], [5.5]]],
        [[[5.5], [3.3, 4.4], [5.5]], [[], [6.6, 7.7, 8.8, 9.9], []]],
    ]
    assert to_list(listarray[:, :2, [0, 1]]) == [
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]],
        [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]],
    ]
    assert to_list(listarray[:1, [0, 0, 1, 1], [0, 1, 0, 1]]) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5]]
    ]
    assert to_list(listarray[:1, [1, 1, 0, 0], [1, 0, 1, 0]]) == [
        [[5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]
    ]
    assert (
        listarray[:1, [1, 1, 0, 0], [1, 0, 1, 0]].form
        == listarray.typetracer[:1, [1, 1, 0, 0], [1, 0, 1, 0]].form
    )


content2 = ak.layout.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(-1, 7))
regulararrayA = ak.layout.RegularArray(content2, 5, zeros_length=0)
regulararrayB = ak.layout.RegularArray(regulararrayA, 3, zeros_length=0)
modelA = np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7)
modelB = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

regulararrayA = v1_to_v2(regulararrayA)
regulararrayB = v1_to_v2(regulararrayB)


def test_numpy():
    assert to_list(regulararrayA) == to_list(modelA)
    assert to_list(regulararrayB) == to_list(modelB)

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 4, -5), depth):
            assert to_list(modelA[cuts]) == to_list(regulararrayA[cuts])
            if depth < 3:
                assert regulararrayA.typetracer[cuts].form == regulararrayA[cuts].form

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(
            (slice(None), slice(1, None), slice(None, -1), slice(None, None, 2)), depth
        ):
            assert to_list(modelA[cuts]) == to_list(regulararrayA[cuts])
            if depth < 3:
                assert regulararrayA.typetracer[cuts].form == regulararrayA[cuts].form

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(
            (slice(1, None), slice(None, -1), 2, -2), depth
        ):
            assert to_list(modelA[cuts]) == to_list(regulararrayA[cuts])
            if depth < 3:
                assert regulararrayA.typetracer[cuts].form == regulararrayA[cuts].form

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(
            ([2, 0, 0, 1], [1, -2, 0, -1], 2, -2), depth
        ):
            assert to_list(modelA[cuts]) == to_list(regulararrayA[cuts])
            if depth < 3:
                assert regulararrayA.typetracer[cuts].form == regulararrayA[cuts].form

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(
            ([2, 0, 0, 1], [1, -2, 0, -1], slice(1, None), slice(None, -1)), depth
        ):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert to_list(modelA[cuts]) == to_list(regulararrayA[cuts])
            if depth < 3:
                assert regulararrayA.typetracer[cuts].form == regulararrayA[cuts].form

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-2, -1, 0, 1, 1), depth):
            assert to_list(modelB[cuts]) == to_list(regulararrayB[cuts])
            if depth < 4:
                assert regulararrayB.typetracer[cuts].form == regulararrayB[cuts].form

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations(
            (-1, 0, 1, slice(1, None), slice(None, -1)), depth
        ):
            assert to_list(modelB[cuts]) == to_list(regulararrayB[cuts])
            if depth < 4:
                assert regulararrayB.typetracer[cuts].form == regulararrayB[cuts].form

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations(
            (-1, 0, [1, 0, 0, 1], [0, 1, -1, 1], slice(None, -1)), depth
        ):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert to_list(modelB[cuts]) == to_list(regulararrayB[cuts])
            if depth < 4:
                assert regulararrayB.typetracer[cuts].form == regulararrayB[cuts].form
