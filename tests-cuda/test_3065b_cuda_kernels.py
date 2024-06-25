from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak
from awkward._behavior import behavior_of
from awkward._nplikes.typetracer import TypeTracer

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0582_propagate_context_in_broadcast_and_apply_firsts():
    array = ak.Array([[[0, 1, 2], []], [[3, 4]], [], [[5], [6, 7, 8, 9]]])
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.firsts(cuda_array, axis=0)) == [[0, 1, 2], []]
    assert to_list(ak.operations.firsts(cuda_array, axis=1)) == [
        [0, 1, 2],
        [3, 4],
        None,
        [5],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=2)) == [
        [0, None],
        [3],
        [],
        [5, 6],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=-1)) == [
        [0, None],
        [3],
        [],
        [5, 6],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=-2)) == [
        [0, 1, 2],
        [3, 4],
        None,
        [5],
    ]
    assert to_list(ak.operations.firsts(cuda_array, axis=-3)) == [
        [0, 1, 2],
        [],
    ]

    with pytest.raises(ValueError):
        ak.operations.firsts(cuda_array, axis=-4)


def test_0582_propagate_context_in_broadcast_and_apply_toregular():
    array = ak.Array(
        [
            {
                "x": np.arange(2 * 3 * 5).reshape(2, 3, 5).tolist(),
                "y": np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7),
            }
        ]
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert str(cuda_array.type) in (
        "1 * {x: var * var * var * int64, y: var * var * var * var * int64}",
        "1 * {y: var * var * var * var * int64, x: var * var * var * int64}",
    )
    assert str(ak.operations.to_regular(cuda_array, axis=-1).type) in (
        "1 * {x: var * var * 5 * int64, y: var * var * var * 7 * int64}",
        "1 * {y: var * var * var * 7 * int64, x: var * var * 5 * int64}",
    )
    assert str(ak.operations.to_regular(cuda_array, axis=-2).type) in (
        "1 * {x: var * 3 * var * int64, y: var * var * 5 * var * int64}",
        "1 * {y: var * var * 5 * var * int64, x: var * 3 * var * int64}",
    )
    assert str(ak.operations.to_regular(cuda_array, axis=-3).type) in (
        "1 * {x: 2 * var * var * int64, y: var * 3 * var * var * int64}",
        "1 * {y: var * 3 * var * var * int64, x: 2 * var * var * int64}",
    )


def test_0582_propagate_context_in_broadcast_and_apply_cartesian():
    one = ak.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist())
    two = ak.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist())

    cuda_one = ak.to_backend(one, "cuda")
    cuda_two = ak.to_backend(two, "cuda")

    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=0, nested=True).type)
        == "2 * 2 * (var * var * var * int64, var * var * var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=1, nested=True).type)
        == "2 * var * var * (var * var * int64, var * var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=2, nested=True).type)
        == "2 * var * var * var * (var * int64, var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=3, nested=True).type)
        == "2 * var * var * var * var * (int64, int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-1, nested=True).type)
        == "2 * var * var * var * var * (int64, int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-2, nested=True).type)
        == "2 * var * var * var * (var * int64, var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-3, nested=True).type)
        == "2 * var * var * (var * var * int64, var * var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-4, nested=True).type)
        == "2 * 2 * (var * var * var * int64, var * var * var * int64)"
    )

    with pytest.raises(ValueError):
        ak.operations.cartesian([cuda_one, cuda_two], axis=-5, nested=True)

    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=0).type)
        == "4 * (var * var * var * int64, var * var * var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=1).type)
        == "2 * var * (var * var * int64, var * var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=2).type)
        == "2 * var * var * (var * int64, var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=3).type)
        == "2 * var * var * var * (int64, int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-1).type)
        == "2 * var * var * var * (int64, int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-2).type)
        == "2 * var * var * (var * int64, var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-3).type)
        == "2 * var * (var * var * int64, var * var * int64)"
    )
    assert (
        str(ak.operations.cartesian([cuda_one, cuda_two], axis=-4).type)
        == "4 * (var * var * var * int64, var * var * var * int64)"
    )

    with pytest.raises(ValueError):
        ak.operations.cartesian([cuda_one, cuda_two], axis=-5)


def test_0193_is_none_axis_parameter():
    array = ak.Array([1, 2, 3, None, 5])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array).to_list() == [
        False,
        False,
        False,
        True,
        False,
    ]

    array = ak.Array([[1, 2, 3], [], [None, 5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array).to_list() == [
        False,
        False,
        False,
    ]
    assert ak.operations.is_none(cuda_array, axis=1).to_list() == [
        [False, False, False],
        [],
        [True, False],
    ]

    array = ak.Array([[1, None, 2, 3], [], [None, 5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array, axis=1).to_list() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.operations.is_none(cuda_array, axis=-1).to_list() == [
        [False, True, False, False],
        [],
        [True, False],
    ]
    assert ak.operations.is_none(cuda_array, axis=-2).to_list() == [
        False,
        False,
        False,
    ]
    with pytest.raises(ValueError):
        ak.operations.is_none(cuda_array, axis=-3)

    array = ak.Array([[1, None, 2, 3], None, [None, 5]])
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.is_none(cuda_array, axis=-2).to_list() == [False, True, False]

    one = ak.operations.from_iter([1, None, 3], highlevel=False)
    two = ak.operations.from_iter([[], [1], None, [3, 3, 3]], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int64))
    array = ak.Array(ak.contents.UnionArray(tags, index, [one, two]), check_valid=True)
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.to_list(cuda_array) == [1, [], [1], None, 3, None, [3, 3, 3]]

    assert ak.to_list(ak.operations.is_none(cuda_array)) == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ]


def test_0493_zeros_ones_full_like():
    array = ak.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
            [],
            [
                {"x": 3.3, "y": [1, 2, None, 3]},
                False,
                False,
                True,
                {"x": 4.4, "y": [1, 2, None, 3, 4]},
            ],
        ]
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.full_like(cuda_array, 12.3).to_list() == [
        [{"x": 12.3, "y": []}, {"x": 12.3, "y": [12]}, {"x": 12.3, "y": [12, 12]}],
        [],
        [
            {"x": 12.3, "y": [12, 12, None, 12]},
            True,
            True,
            True,
            {"x": 12.3, "y": [12, 12, None, 12, 12]},
        ],
    ]

    assert ak.operations.zeros_like(cuda_array).to_list() == [
        [{"x": 0.0, "y": []}, {"x": 0.0, "y": [0]}, {"x": 0.0, "y": [0, 0]}],
        [],
        [
            {"x": 0.0, "y": [0, 0, None, 0]},
            False,
            False,
            False,
            {"x": 0.0, "y": [0, 0, None, 0, 0]},
        ],
    ]

    assert ak.operations.ones_like(cuda_array).to_list() == [
        [{"x": 1.0, "y": []}, {"x": 1.0, "y": [1]}, {"x": 1.0, "y": [1, 1]}],
        [],
        [
            {"x": 1.0, "y": [1, 1, None, 1]},
            True,
            True,
            True,
            {"x": 1.0, "y": [1, 1, None, 1, 1]},
        ],
    ]


def test_0496_provide_local_index():
    array = ak.highlevel.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.local_index(cuda_array, axis=0)) == [
        0,
        1,
        2,
        3,
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=1)) == [
        [0, 1],
        [0],
        [],
        [0, 1, 2],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=2)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=-1)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=-2)) == [
        [0, 1],
        [0],
        [],
        [0, 1, 2],
    ]
    assert to_list(ak.operations.local_index(cuda_array, axis=-3)) == [
        0,
        1,
        2,
        3,
    ]

    assert to_list(
        ak.operations.zip(
            [
                ak.operations.local_index(cuda_array, axis=0),
                ak.operations.local_index(cuda_array, axis=1),
                ak.operations.local_index(cuda_array, axis=2),
            ]
        )
    ) == [
        [[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []],
        [[(1, 0, 0), (1, 0, 1)]],
        [],
        [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]],
    ]


def test_0504_block_ufuncs_for_strings():
    def _apply_ufunc(ufunc, method, inputs, kwargs):
        nextinputs = []
        for x in inputs:
            if (
                isinstance(x, ak.highlevel.Array)
                and x.layout.is_indexed
                and not x.layout.is_option
            ):
                nextinputs.append(
                    ak.highlevel.Array(x.layout.project(), behavior=behavior_of(x))
                )
            else:
                nextinputs.append(x)

        return getattr(ufunc, method)(*nextinputs, **kwargs)

    behavior = {}
    behavior[np.ufunc, "categorical"] = _apply_ufunc

    array = ak.highlevel.Array(
        ak.contents.IndexedArray(
            ak.index.Index64(np.array([0, 1, 2, 1, 3, 1, 4])),
            ak.contents.NumpyArray(np.array([321, 1.1, 123, 999, 2])),
            parameters={"__array__": "categorical"},
        ),
        behavior=behavior,
    )
    cuda_array = ak.to_backend(array, "cuda")

    assert to_list(cuda_array * 10) == [3210, 11, 1230, 11, 9990, 11, 20]

    cuda_array = ak.highlevel.Array(["HAL"])

    with pytest.raises(TypeError):
        cuda_array + 1


def test_0527_fix_unionarray_ufuncs_and_parameters_in_merging_0459():
    plain_plain = ak.highlevel.Array([[0.0, 1.1, 2.2, 3.3, 4.4]])
    cuda_plain_plain = ak.to_backend(plain_plain, "cuda")

    cuda_array_plain = ak.operations.with_parameter(
        cuda_plain_plain, "__list__", "zoinks"
    )
    cuda_plain_isdoc = ak.operations.with_parameter(
        cuda_plain_plain, "__doc__", "This is a zoink."
    )
    cuda_array_isdoc = ak.operations.with_parameter(
        cuda_array_plain, "__doc__", "This is a zoink."
    )

    assert ak.operations.parameters(cuda_plain_plain) == {}
    assert ak.operations.parameters(cuda_array_plain) == {"__list__": "zoinks"}
    assert ak.operations.parameters(cuda_plain_isdoc) == {"__doc__": "This is a zoink."}
    assert ak.operations.parameters(cuda_array_isdoc) == {
        "__list__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert (
        ak.operations.parameters(
            ak.operations.concatenate([cuda_plain_plain, cuda_plain_plain])
        )
        == {}
    )
    assert ak.operations.parameters(
        ak.operations.concatenate([cuda_array_plain, cuda_array_plain])
    ) == {"__list__": "zoinks"}
    assert ak.operations.parameters(
        ak.operations.concatenate([cuda_plain_isdoc, cuda_plain_isdoc])
    ) == {"__doc__": "This is a zoink."}
    assert ak.operations.parameters(
        ak.operations.concatenate([cuda_array_isdoc, cuda_array_isdoc])
    ) == {
        "__list__": "zoinks",
        "__doc__": "This is a zoink.",
    }

    assert isinstance(
        ak.operations.concatenate([cuda_plain_plain, cuda_plain_plain]).layout,
        ak.contents.ListOffsetArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_array_plain, cuda_array_plain]).layout,
        ak.contents.ListOffsetArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_plain_isdoc, cuda_plain_isdoc]).layout,
        ak.contents.ListOffsetArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_array_isdoc, cuda_array_isdoc]).layout,
        ak.contents.ListOffsetArray,
    )

    assert (
        ak.operations.parameters(
            ak.operations.concatenate([cuda_plain_plain, cuda_array_plain])
        )
        == {}
    )
    assert (
        ak.operations.parameters(
            ak.operations.concatenate([cuda_plain_isdoc, cuda_array_isdoc])
        )
        == {}
    )
    assert (
        ak.operations.parameters(
            ak.operations.concatenate([cuda_array_plain, cuda_plain_plain])
        )
        == {}
    )
    assert (
        ak.operations.parameters(
            ak.operations.concatenate([cuda_array_isdoc, cuda_plain_isdoc])
        )
        == {}
    )

    assert isinstance(
        ak.operations.concatenate([cuda_plain_plain, cuda_array_plain]).layout,
        ak.contents.UnionArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_plain_isdoc, cuda_array_isdoc]).layout,
        ak.contents.UnionArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_array_plain, cuda_plain_plain]).layout,
        ak.contents.UnionArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_array_isdoc, cuda_plain_isdoc]).layout,
        ak.contents.UnionArray,
    )

    assert (
        ak.operations.parameters(
            ak.operations.concatenate([cuda_plain_plain, cuda_plain_isdoc])
        )
        == {}
    )
    assert ak.operations.parameters(
        ak.operations.concatenate([cuda_array_plain, cuda_array_isdoc])
    ) == {"__list__": "zoinks"}
    assert (
        ak.operations.parameters(
            ak.operations.concatenate([cuda_plain_isdoc, cuda_plain_plain])
        )
        == {}
    )
    assert ak.operations.parameters(
        ak.operations.concatenate([cuda_array_isdoc, cuda_array_plain])
    ) == {"__list__": "zoinks"}

    assert isinstance(
        ak.operations.concatenate([cuda_plain_plain, cuda_plain_isdoc]).layout,
        ak.contents.ListOffsetArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_array_plain, cuda_array_isdoc]).layout,
        ak.contents.ListOffsetArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_plain_isdoc, cuda_plain_plain]).layout,
        ak.contents.ListOffsetArray,
    )
    assert isinstance(
        ak.operations.concatenate([cuda_array_isdoc, cuda_array_plain]).layout,
        ak.contents.ListOffsetArray,
    )


def test_0527_fix_unionarray_ufuncs_and_parameters_in_merging_0522():
    content1 = ak.highlevel.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    content2 = ak.highlevel.Array([[0], [100], [200], [300], [400]]).layout
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1], np.int8))
    index = ak.index.Index64(np.array([0, 1, 2, 0, 1, 3, 4, 2, 3, 4], np.int64))
    unionarray = ak.highlevel.Array(
        ak.contents.UnionArray(tags, index, [content1, content2])
    )
    cuda_unionarray = ak.to_backend(unionarray, "cuda")

    assert cuda_unionarray.to_list() == [
        0.0,
        1.1,
        2.2,
        [0],
        [100],
        3.3,
        4.4,
        [200],
        [300],
        [400],
    ]

    assert (cuda_unionarray + 10).to_list() == [
        10.0,
        11.1,
        12.2,
        [10],
        [110],
        13.3,
        14.4,
        [210],
        [310],
        [410],
    ]
    assert (10 + cuda_unionarray).to_list() == [
        10.0,
        11.1,
        12.2,
        [10],
        [110],
        13.3,
        14.4,
        [210],
        [310],
        [410],
    ]

    assert (cuda_unionarray + range(0, 100, 10)).to_list() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]
    assert (range(0, 100, 10) + cuda_unionarray).to_list() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]

    assert (cuda_unionarray + cp.arange(0, 100, 10)).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]
    assert (cp.arange(0, 100, 10) + cuda_unionarray).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]

    assert (cuda_unionarray + ak.highlevel.Array(cp.arange(0, 100, 10))).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]
    assert (ak.highlevel.Array(cp.arange(0, 100, 10)) + cuda_unionarray).tolist() == [
        0.0,
        11.1,
        22.2,
        [30],
        [140],
        53.3,
        64.4,
        [270],
        [380],
        [490],
    ]

    # assert (cuda_unionarray + cuda_unionarray).to_list() == [
    #     0.0,
    #     2.2,
    #     4.4,
    #     [0],
    #     [200],
    #     6.6,
    #     8.8,
    #     [400],
    #     [600],
    #     [800],
    # ]


def test_0959_getitem_array_implementation_UnionArray_NumpyArray():
    v2a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.from_iter([[1], [2], [3]], highlevel=False),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1, 3], cp.int64)]
    assert to_list(cuda_resultv2) == [5.5, 4.4, [2]]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1, 3], cp.int64)].form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_BitMaskedArray_NumpyArray():
    v2a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1, 4], cp.int64)]
    assert to_list(cuda_resultv2) == [0.0, 1.0, None]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1, 4], cp.int64)].form
        == cuda_resultv2.form
    )

    v2b = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    cuda_v2b = ak.to_backend(v2b, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2b[cp.array([0, 1, 4], cp.int64)]
    assert to_list(cuda_resultv2) == [0.0, 1.0, None]
    assert (
        cuda_v2b.to_typetracer()[cp.array([0, 1, 4], cp.int64)].form
        == cuda_resultv2.form
    )

    v2c = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    cuda_v2c = ak.to_backend(v2c, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2c[cp.array([0, 1, 4], cp.int64)]
    assert to_list(cuda_resultv2) == [0.0, 1.0, None]
    assert (
        cuda_v2c.to_typetracer()[cp.array([0, 1, 4], cp.int64)].form
        == cuda_resultv2.form
    )

    v2d = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    cuda_v2d = ak.to_backend(v2d, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2d[cp.array([0, 1, 4], cp.int64)]
    assert to_list(cuda_resultv2) == [0.0, 1.0, None]
    assert (
        cuda_v2d.to_typetracer()[cp.array([0, 1, 4], cp.int64)].form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_BitMaskedArray_RecordArray_NumpyArray():
    v2a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)
    indexa = ak.index.Index(cp.array([0, 1, 4], cp.int64))
    cuda_resultv2 = cuda_v2a._carry(indexa, False)
    assert to_list(cuda_resultv2) == [{"nest": 0.0}, {"nest": 1.0}, None]
    assert (
        cuda_v2a.to_typetracer()
        ._carry(indexa.to_nplike(TypeTracer.instance()), False)
        .form
        == cuda_resultv2.form
    )

    v2b = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    cuda_v2b = ak.to_backend(v2b, "cuda", highlevel=False)

    indexb = ak.index.Index(cp.array([1, 1, 4], cp.int64))
    cuda_resultv2 = cuda_v2b._carry(indexb, False)
    assert to_list(cuda_resultv2) == [{"nest": 1.0}, {"nest": 1.0}, None]
    assert (
        cuda_v2b.to_typetracer()
        ._carry(indexb.to_nplike(TypeTracer.instance()), False)
        .form
        == cuda_resultv2.form
    )

    v2c = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    cuda_v2c = ak.to_backend(v2c, "cuda", highlevel=False)

    indexc = ak.index.Index(cp.array([0, 1, 4], cp.int64))
    cuda_resultv2 = cuda_v2c._carry(indexc, False)
    assert to_list(cuda_resultv2) == [{"nest": 0.0}, {"nest": 1.0}, None]
    assert (
        cuda_v2c.to_typetracer()
        ._carry(indexc.to_nplike(TypeTracer.instance()), False)
        .form
        == cuda_resultv2.form
    )

    v2d = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    cuda_v2d = ak.to_backend(v2d, "cuda", highlevel=False)

    indexd = ak.index.Index(cp.array([0, 0, 0], cp.int64))
    cuda_resultv2 = cuda_v2d._carry(indexd, False)
    assert to_list(cuda_resultv2) == [{"nest": 0.0}, {"nest": 0.0}, {"nest": 0.0}]
    assert (
        cuda_v2d.to_typetracer()
        ._carry(indexd.to_nplike(TypeTracer.instance()), False)
        .form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_IndexedArray_RecordArray_NumpyArray():
    v2a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1, 4], cp.int64)]
    assert to_list(cuda_resultv2) == [{"nest": 3.3}, {"nest": 3.3}, {"nest": 5.5}]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1, 4], cp.int64)].form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_IndexedOptionArray_RecordArray_NumpyArray():
    v2a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1, 4], cp.int64)]
    assert to_list(cuda_resultv2) == [{"nest": 3.3}, {"nest": 3.3}, None]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1, 4], cp.int64)].form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_ByteMaskedArray_RecordArray_NumpyArray():
    v2a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    index = ak.index.Index(cp.array([0, 1, 4], cp.int64))
    cuda_resultv2 = cuda_v2a._carry(index, False)
    assert to_list(cuda_resultv2) == [{"nest": 1.1}, None, {"nest": 5.5}]
    assert (
        cuda_v2a.to_typetracer()
        ._carry(index.to_nplike(TypeTracer.instance()), False)
        .form
        == cuda_resultv2.form
    )

    v2b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=False,
    )
    cuda_v2b = ak.to_backend(v2b, "cuda", highlevel=False)

    indexb = ak.index.Index(cp.array([3, 1, 4], cp.int64))
    cuda_resultv2 = cuda_v2b._carry(indexb, False)
    assert to_list(cuda_resultv2) == [None, None, {"nest": 5.5}]
    assert (
        cuda_v2b.to_typetracer()
        ._carry(indexb.to_nplike(TypeTracer.instance()), False)
        .form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_IndexedArray_NumpyArray():
    v2a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1, 4], cp.int64)]
    assert to_list(cuda_resultv2) == [3.3, 3.3, 5.5]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1, 4], cp.int64)].form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1, -1], cp.int64)]
    assert to_list(cuda_resultv2) == [3.3, 3.3, 5.5]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1, -1], cp.int64)].form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_RecordArray_NumpyArray():
    v2a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([1, 2], cp.int64)]
    assert to_list(cuda_resultv2) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]
    assert (
        cuda_v2a.to_typetracer()[cp.array([1, 2], cp.int64)].form == cuda_resultv2.form
    )

    v2b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )
    cuda_v2b = ak.to_backend(v2b, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2b[cp.array([0, 1, 2, 3, -1], cp.int64)]
    assert to_list(cuda_resultv2) == [(0, 0.0), (1, 1.1), (2, 2.2), (3, 3.3), (4, 4.4)]
    assert (
        cuda_v2b.to_typetracer()[cp.array([0, 1, 2, 3, -1], cp.int64)].form
        == cuda_resultv2.form
    )

    v2c = ak.contents.recordarray.RecordArray([], [], 10)
    cuda_v2c = ak.to_backend(v2c, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2c[cp.array([0], cp.int64)]
    assert to_list(cuda_resultv2) == [{}]
    assert cuda_v2c.to_typetracer()[cp.array([0], cp.int64)].form == cuda_resultv2.form

    v2d = ak.contents.recordarray.RecordArray([], None, 10)
    cuda_v2d = ak.to_backend(v2d, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2d[cp.array([0], cp.int64)]
    assert to_list(cuda_resultv2) == [()]
    assert cuda_v2d.to_typetracer()[cp.array([0], cp.int64)].form == cuda_resultv2.form


def test_0959_getitem_array_implementation_RegularArray_NumpyArray():
    v2a = ak.contents.regulararray.RegularArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1], cp.int64)]
    assert to_list(cuda_resultv2) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1], cp.int64)].form == cuda_resultv2.form
    )

    v2b = ak.contents.regulararray.RegularArray(
        ak.contents.emptyarray.EmptyArray(), 0, zeros_length=10
    )
    cuda_v2b = ak.to_backend(v2b, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2b[cp.array([0, 0, 0], cp.int64)]
    assert to_list(cuda_resultv2) == [[], [], []]
    assert (
        cuda_v2b.to_typetracer()[cp.array([0, 0, 0], cp.int64)].form
        == cuda_resultv2.form
    )

    assert to_list(cuda_resultv2) == [[], [], []]


def test_0959_getitem_array_implementation_ListArray_NumpyArray():
    v2a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([1, -1], cp.int64)]
    assert to_list(cuda_resultv2) == [[], [4.4, 5.5]]
    assert (
        cuda_v2a.to_typetracer()[cp.array([1, -1], cp.int64)].form == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_UnionArray_RecordArray_NumpyArray():
    v2a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.contents.recordarray.RecordArray(
                [ak.from_iter([[1], [2], [3]], highlevel=False)],
                ["nest"],
            ),
            ak.contents.recordarray.RecordArray(
                [
                    ak.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1, 1], cp.int64)]
    assert to_list(cuda_resultv2) == [{"nest": 5.5}, {"nest": 4.4}, {"nest": 4.4}]
    assert (
        cuda_v2a.to_typetracer()[cp.array([0, 1, 1], cp.int64)].form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_RecordArray_NumpyArray_lazy():
    v2a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a._carry(ak.index.Index(cp.array([1, 2], cp.int64)), True)
    assert to_list(cuda_resultv2) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]
    assert (
        cuda_v2a.to_typetracer()
        ._carry(ak.index.Index(cp.array([1, 2], cp.int64)), True)
        .form
        == cuda_resultv2.form
    )

    v2b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )
    cuda_v2b = ak.to_backend(v2b, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2b._carry(
        ak.index.Index(cp.array([0, 1, 2, 3, 4], cp.int64)), True
    )
    assert to_list(cuda_resultv2) == [(0, 0.0), (1, 1.1), (2, 2.2), (3, 3.3), (4, 4.4)]
    assert (
        cuda_v2b.to_typetracer()
        ._carry(ak.index.Index(cp.array([0, 1, 2, 3, 4], cp.int64)), True)
        .form
        == cuda_resultv2.form
    )

    v2c = ak.contents.recordarray.RecordArray([], [], 10)
    cuda_v2c = ak.to_backend(v2c, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2c[cp.array([0], cp.int64)]
    assert to_list(cuda_resultv2) == [{}]
    assert cuda_v2c.to_typetracer()[cp.array([0], cp.int64)].form == cuda_resultv2.form

    v2d = ak.contents.recordarray.RecordArray([], None, 10)
    cuda_v2d = ak.to_backend(v2d, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2d[cp.array([0], cp.int64)]
    assert to_list(cuda_resultv2) == [()]
    assert cuda_v2d.to_typetracer()[cp.array([0], cp.int64)].form == cuda_resultv2.form


def test_0959_getitem_array_implementation_RegularArray_RecordArray_NumpyArray():
    v2a = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a._carry(ak.index.Index(cp.array([0], cp.int64)), False)
    assert to_list(cuda_resultv2) == [[{"nest": 0.0}, {"nest": 1.1}, {"nest": 2.2}]]
    assert (
        cuda_v2a.to_typetracer()
        ._carry(ak.index.Index(cp.array([0], cp.int64)), False)
        .form
        == cuda_resultv2.form
    )

    v2b = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )
    cuda_v2b = ak.to_backend(v2b, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2b._carry(ak.index.Index(cp.array([0], cp.int64)), False)
    assert to_list(cuda_resultv2) == [[]]
    assert (
        cuda_v2b.to_typetracer()
        ._carry(ak.index.Index(cp.array([0], cp.int64)), False)
        .form
        == cuda_resultv2.form
    )


def test_0959_getitem_array_implementation_ListArray_RecordArray_NumpyArray():
    v2a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )
    cuda_v2a = ak.to_backend(v2a, "cuda", highlevel=False)

    cuda_resultv2 = cuda_v2a[cp.array([0, 1], np.int64)]
    assert to_list(cuda_resultv2) == [[{"nest": 1.1}, {"nest": 2.2}, {"nest": 3.3}], []]
    assert (
        cuda_v2a.to_typetracer()[np.array([0, 1], np.int64)].form == cuda_resultv2.form
    )
