# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

to_jaggedtensor = ak.operations.to_jaggedtensor

torch = pytest.importorskip("torch")
fbgemm_gpu = pytest.importorskip("fbgemm_gpu")

content = ak.contents.NumpyArray(
    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
)
starts1 = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
stops1 = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
starts2 = ak.index.Index64(np.array([0, 3]))
stops2 = ak.index.Index64(np.array([3, 5]))

array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
content2 = ak.contents.NumpyArray(array.reshape(-1))
inneroffsets = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
outeroffsets = ak.index.Index64(np.array([0, 3, 6]))


def to_float32(array, highlevel=False):
    return ak.values_astype(array, np.float32, highlevel=highlevel)


def test_convert_to_jaggedtensor():
    # a test for ListArray -> JaggedTensor
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array1 = to_float32(array1)
    jagged1 = to_jaggedtensor(array1)
    assert torch.equal(
        jagged1[0], torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    assert torch.equal(jagged1[1][0], torch.tensor([0, 3, 3, 5, 6, 9]))

    # a test for NumpyArray -> JaggedTensor
    array2 = content
    assert torch.equal(
        to_jaggedtensor(array2),
        torch.tensor(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=torch.float64
        ),
    )

    # a test for RegularArray -> JaggedTensor
    array3 = ak.contents.RegularArray(content, size=2)
    array3 = to_float32(array3)
    assert torch.equal(
        to_jaggedtensor(array3),
        torch.tensor(
            [
                [1.1, 2.2],
                [3.3, 4.4],
                [5.5, 6.6],
                [7.7, 8.8],
            ]
        ),
    )

    # try a single line awkward array
    array4 = ak.Array([3, 1, 4, 1, 9, 2, 6])
    assert torch.equal(to_jaggedtensor(array4), torch.tensor([3, 1, 4, 1, 9, 2, 6]))

    # try a multiple ragged array
    array5 = ak.Array([[[1.1, 2.2], [3.3]], [], [[4.4, 5.5]]])
    array5 = to_float32(array5, highlevel=True)
    jagged2 = to_jaggedtensor(array5)
    assert torch.equal(
        jagged2[0], torch.tensor([1.1000, 2.2000, 3.3000, 4.4000, 5.5000])
    )
    assert torch.equal(jagged2[1][0], torch.tensor([0, 2, 2, 3]))
    assert torch.equal(jagged2[1][1], torch.tensor([0, 2, 3, 5]))

    # try a listoffset array inside a listoffset array
    array6 = ak.contents.ListOffsetArray(
        outeroffsets, ak.contents.ListOffsetArray(inneroffsets, content2)
    )
    jagged3 = to_jaggedtensor(array6)
    assert torch.equal(
        jagged3[0],
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
            ]
        ),
    )
    assert torch.equal(jagged3[1][0], torch.tensor([0, 3, 6]))
    assert torch.equal(jagged3[1][1], torch.tensor([0, 5, 10, 15, 20, 25, 30]))

    # try a list array inside a list array
    array7 = ak.contents.ListArray(
        starts2, stops2, ak.contents.ListArray(starts1, stops1, content)
    )
    array7 = to_float32(array7)
    jagged4 = to_jaggedtensor(array7)
    assert torch.equal(
        jagged4[0], torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    assert torch.equal(jagged4[1][0], torch.tensor([0, 3, 5]))
    assert torch.equal(jagged4[1][1], torch.tensor([0, 3, 3, 5, 6, 9]))

    # try just a python list
    array8 = [3, 1, 4, 1, 9, 2, 6]
    assert torch.equal(to_jaggedtensor(array8), torch.tensor([3, 1, 4, 1, 9, 2, 6]))

    # try array with three inner dimensions
    array9 = ak.Array([[[[1.1, 2.2], [3.3]], [[4.4]]], [], [[[5.5, 6.6], [7.7]]]])
    array9 = to_float32(array9, highlevel=True)
    jagged5 = to_jaggedtensor(array9)
    assert torch.equal(jagged5[0], torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]))
    # whole offset list loooks like this -> [tensor([0, 2, 2, 3]), tensor([0, 2, 3, 5]), tensor([0, 2, 3, 4, 6, 7])]
    assert torch.equal(jagged5[1][0], torch.tensor([0, 2, 2, 3]))
    assert torch.equal(jagged5[1][1], torch.tensor([0, 2, 3, 5]))
    assert torch.equal(jagged5[1][2], torch.tensor([0, 2, 3, 4, 6, 7]))


def test_regular_array():
    # try to keep the regular arrays if possible:
    array10 = ak.Array([[[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], [[7.7, 8.8], [9.9, 10]]])
    array10 = to_float32(array10, highlevel=True)
    regular1 = ak.to_regular(array10, axis=2)
    jagged6 = to_jaggedtensor(regular1)
    assert torch.equal(
        jagged6[0],
        torch.tensor([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8], [9.9, 10.0]]),
    )
    assert torch.equal(jagged6[1][0], torch.tensor([0, 3, 5]))

    # otherwise (if RegularArray contains ListArray or ListOffsetArray) convert to padded tensor (if keep_regular = False)
    array11 = ak.Array([[[1.1, 2.2], [3.3, 4.4, 5.5, 6.6]], [[7.7, 8.8], [9.9, 10]]])
    array11 = to_float32(array11, highlevel=True)
    regular2 = ak.to_regular(array11, axis=1)
    jagged7 = to_jaggedtensor(regular2, keep_regular=False)
    assert torch.equal(
        jagged7,
        torch.tensor(
            [
                [[1.1, 2.2, 0.0, 0.0], [3.3, 4.4, 5.5, 6.6]],
                [[7.7, 8.8, 0.0, 0.0], [9.9, 10.0, 0.0, 0.0]],
            ]
        ),
    )
