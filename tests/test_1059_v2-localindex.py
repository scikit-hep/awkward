# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_listoffsetarray_localindex():
    v1_array = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3, 4]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [
        [0, 1, 2],
        [],
        [0, 1],
        [0],
        [0, 1, 2, 3],
    ]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(-1)) == [
        [0, 1, 2],
        [],
        [0, 1],
        [0],
        [0, 1, 2, 3],
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [0, 1, 2, 3, 4]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form

    with pytest.raises(IndexError):
        v2_array.localindex(-3)
    with pytest.raises(IndexError):
        v2_array.localindex(2)

    v1_array = ak.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[6.6, 7.7, 8.8, 9.9]]],
        highlevel=False,
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [[0, 1, 2], [], [0], [0]]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(2)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        [[0]],
        [[0, 1, 2, 3]],
    ]
    assert v2_array.typetracer.localindex(2).form == v2_array.localindex(2).form
    assert ak.to_list(v2_array.localindex(-1)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        [[0]],
        [[0, 1, 2, 3]],
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [[0, 1, 2], [], [0], [0]]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form
    assert ak.to_list(v2_array.localindex(-3)) == [0, 1, 2, 3]
    assert v2_array.typetracer.localindex(-3).form == v2_array.localindex(-3).form

    with pytest.raises(IndexError):
        v2_array.localindex(-4)
    with pytest.raises(IndexError):
        v2_array.localindex(3)


def test_regulararray_localindex():
    v1_array = ak.from_numpy(
        np.arange(2 * 3 * 5).reshape(2, 3, 5), regulararray=True, highlevel=False
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.localindex(0)) == [0, 1]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [[0, 1, 2], [0, 1, 2]]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(2)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert v2_array.typetracer.localindex(2).form == v2_array.localindex(2).form
    assert ak.to_list(v2_array.localindex(-1)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [[0, 1, 2], [0, 1, 2]]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form
    assert ak.to_list(v2_array.localindex(-3)) == [0, 1]
    assert v2_array.typetracer.localindex(-3).form == v2_array.localindex(-3).form

    with pytest.raises(IndexError):
        v2_array.localindex(-4)
    with pytest.raises(IndexError):
        v2_array.localindex(3)

    v1_array = ak.from_numpy(
        np.arange(2 * 3 * 5 * 10).reshape(2, 3, 5, 10),
        regulararray=True,
        highlevel=False,
    )
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.localindex(0)) == [0, 1]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [[0, 1, 2], [0, 1, 2]]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(2)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert v2_array.typetracer.localindex(2).form == v2_array.localindex(2).form
    assert ak.to_list(v2_array.localindex(3)) == [
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
    ]
    assert v2_array.typetracer.localindex(3).form == v2_array.localindex(3).form
    assert ak.to_list(v2_array.localindex(-1)) == [
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
        [
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
        ],
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
    ]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form
    assert ak.to_list(v2_array.localindex(-3)) == [[0, 1, 2], [0, 1, 2]]
    assert v2_array.typetracer.localindex(-3).form == v2_array.localindex(-3).form
    assert ak.to_list(v2_array.localindex(-4)) == [0, 1]
    assert v2_array.typetracer.localindex(-4).form == v2_array.localindex(-4).form

    with pytest.raises(IndexError):
        v2_array.localindex(-5)
    with pytest.raises(IndexError):
        v2_array.localindex(4)

    v1_array = ak.Array(
        ak.layout.RegularArray(
            ak.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
        )
    ).layout
    v2_array = v1_to_v2(v1_array)
    assert ak.to_list(v2_array.localindex(0)) == []
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == []
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(2)) == []
    assert v2_array.typetracer.localindex(2).form == v2_array.localindex(2).form
    assert ak.to_list(v2_array.localindex(-1)) == []
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == []
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form
    assert ak.to_list(v2_array.localindex(-3)) == []
    assert v2_array.typetracer.localindex(-3).form == v2_array.localindex(-3).form

    with pytest.raises(IndexError):
        v2_array.localindex(-4)
    with pytest.raises(IndexError):
        v2_array.localindex(3)


def test_bytemaskedarray_localindex():
    content = ak.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v1_array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    v2_array = v1_to_v2(v1_array)

    assert ak.to_list(v1_array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(v2_array.localindex(axis=0)) == [0, 1, 2, 3, 4]
    assert (
        v2_array.typetracer.localindex(axis=0).form == v2_array.localindex(axis=0).form
    )
    assert ak.to_list(v2_array.localindex(axis=-3)) == [0, 1, 2, 3, 4]
    assert (
        v2_array.typetracer.localindex(axis=-3).form
        == v2_array.localindex(axis=-3).form
    )
    assert ak.to_list(v2_array.localindex(axis=1)) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1],
    ]
    assert (
        v2_array.typetracer.localindex(axis=1).form == v2_array.localindex(axis=1).form
    )
    assert ak.to_list(v2_array.localindex(axis=-2)) == [
        [0, 1, 2],
        [],
        None,
        None,
        [0, 1],
    ]
    assert (
        v2_array.typetracer.localindex(axis=-2).form
        == v2_array.localindex(axis=-2).form
    )
    assert ak.to_list(v2_array.localindex(axis=2)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]
    assert (
        v2_array.typetracer.localindex(axis=2).form == v2_array.localindex(axis=2).form
    )
    assert ak.to_list(v2_array.localindex(axis=-1)) == [
        [[0, 1, 2], [], [0, 1]],
        [],
        None,
        None,
        [[], [0, 1, 2]],
    ]
    assert (
        v2_array.typetracer.localindex(axis=-1).form
        == v2_array.localindex(axis=-1).form
    )

    with pytest.raises(IndexError):
        v2_array.localindex(axis=4)
    with pytest.raises(IndexError):
        v2_array.localindex(axis=-4)


def test_numpyarray_localindex():
    v2_array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
    )
    assert ak.to_list(v2_array.localindex(axis=0)) == [0, 1, 2, 3]
    assert (
        v2_array.typetracer.localindex(axis=0).form == v2_array.localindex(axis=0).form
    )
    assert ak.to_list(v2_array.localindex(axis=-1)) == [0, 1, 2, 3]
    assert (
        v2_array.typetracer.localindex(axis=-1).form
        == v2_array.localindex(axis=-1).form
    )

    with pytest.raises(IndexError):
        v2_array.localindex(axis=1)
    with pytest.raises(IndexError):
        v2_array.localindex(axis=2)
    with pytest.raises(IndexError):
        v2_array.localindex(axis=-2)


def test_bitmaskedarray_localindex():
    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert ak.to_list(v2_array.localindex(-1)) == [
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
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )

    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [
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
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [
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
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)


def test_unmaskedarray_localindex():
    v2_array = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )
    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)


def test_unionarray_localindex():
    v2_array = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)


def test_recordarray_localindex():
    v2_array = ak._v2.contents.regulararray.RegularArray(  # noqa: F841
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    assert ak.to_list(v2_array.localindex(0)) == [0, 1]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [[0, 1, 2], [0, 1, 2]]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(-1)) == [[0, 1, 2], [0, 1, 2]]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [0, 1]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form

    with pytest.raises(IndexError):
        v2_array.localindex(-3)
    with pytest.raises(IndexError):
        v2_array.localindex(2)

    v2_array = ak._v2.contents.regulararray.RegularArray(  # noqa: F841
        ak._v2.contents.recordarray.RecordArray(
            [ak._v2.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )

    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(-1)) == [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [
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
    ]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form

    with pytest.raises(IndexError):
        v2_array.localindex(-3)

    v2_array = ak._v2.contents.listarray.ListArray(  # noqa: F841
        ak._v2.index.Index(np.array([4, 100, 1])),
        ak._v2.index.Index(np.array([7, 100, 3, 200])),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )

    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [[0, 1, 2], [], [0, 1]]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(-1)) == [[0, 1, 2], [], [0, 1]]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [0, 1, 2]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form

    with pytest.raises(IndexError):
        v2_array.localindex(-3)
    with pytest.raises(IndexError):
        v2_array.localindex(2)

    v2_array = ak._v2.contents.listoffsetarray.ListOffsetArray(  # noqa: F841
        ak._v2.index.Index(np.array([1, 4, 4, 6])),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    [6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]
                )
            ],
            ["nest"],
        ),
    )
    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(1)) == [[0, 1, 2], [], [0, 1]]
    assert v2_array.typetracer.localindex(1).form == v2_array.localindex(1).form
    assert ak.to_list(v2_array.localindex(-1)) == [[0, 1, 2], [], [0, 1]]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    assert ak.to_list(v2_array.localindex(-2)) == [0, 1, 2]
    assert v2_array.typetracer.localindex(-2).form == v2_array.localindex(-2).form

    with pytest.raises(IndexError):
        v2_array.localindex(-3)
    with pytest.raises(IndexError):
        v2_array.localindex(2)

    v2_array = ak._v2.contents.indexedarray.IndexedArray(  # noqa: F841
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.indexedoptionarray.IndexedOptionArray(  # noqa: F841
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4])),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bytemaskedarray.ByteMaskedArray(  # noqa: F841
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )

    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3, 4]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3, 4]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bytemaskedarray.ByteMaskedArray(  # noqa: F841
        ak._v2.index.Index(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=False,
    )

    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3, 4]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3, 4]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
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
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
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

    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [
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
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form
    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(  # noqa: F841
        ak._v2.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
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

    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [
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
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(  # noqa: F841
        ak._v2.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
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

    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [
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
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(  # noqa: F841
        ak._v2.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
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

    assert ak.to_list(v2_array.localindex(0)) == [
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
    ]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [
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
    ]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.unmaskedarray.UnmaskedArray(  # noqa: F841
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
                )
            ],
            ["nest"],
        )
    )

    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)

    v2_array = ak._v2.contents.unionarray.UnionArray(  # noqa: F841
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak._v2.contents.recordarray.RecordArray(
                [ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3]))], ["nest"]
            ),
            ak._v2.contents.recordarray.RecordArray(
                [
                    ak._v2.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )

    assert ak.to_list(v2_array.localindex(0)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(0).form == v2_array.localindex(0).form
    assert ak.to_list(v2_array.localindex(-1)) == [0, 1, 2, 3, 4, 5, 6]
    assert v2_array.typetracer.localindex(-1).form == v2_array.localindex(-1).form

    with pytest.raises(IndexError):
        v2_array.localindex(-2)
    with pytest.raises(IndexError):
        v2_array.localindex(1)
