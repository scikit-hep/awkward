# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.shape import unknown_length


def test_BitMaskedArray():
    content = ak.contents.NumpyArray(np.arange(13))
    mask = ak.index.IndexU8(np.array([58, 59], dtype=np.uint8))
    array = ak.contents.BitMaskedArray(
        mask,
        content,
        valid_when=False,
        length=unknown_length,
        lsb_order=False,
        length_generator=lambda: 13,
    )
    assert array._length is unknown_length
    assert array.length == 13
    assert array._length == 13
    assert array.to_list() == [
        0,
        1,
        None,
        None,
        None,
        5,
        None,
        7,
        8,
        9,
        None,
        None,
        None,
    ]

    simplified = array.simplified(
        mask,
        content,
        valid_when=False,
        length=unknown_length,
        lsb_order=False,
        length_generator=lambda: 13,
    )
    assert simplified._length is unknown_length
    assert simplified.length == 13
    assert simplified._length == 13
    assert simplified.to_list() == [
        0,
        1,
        None,
        None,
        None,
        5,
        None,
        7,
        8,
        9,
        None,
        None,
        None,
    ]

    array = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 5, 10, 13], dtype=np.int64)),
        ak.contents.BitMaskedArray(
            mask,
            content,
            valid_when=False,
            length=13,
            lsb_order=False,
        ),
    )
    form, length, buffers = ak.to_buffers(array)
    container = {key: (lambda b=buffer: b) for key, buffer in buffers.items()}
    array = ak.from_buffers(form, length, container, highlevel=False)
    assert isinstance(array.content, ak.contents.BitMaskedArray)
    assert array.content._length is unknown_length
    assert array.content._length_generator is not None
    assert array.content.length == 13
    assert array.content._length == 13
    assert array.content._length_generator is None
    assert array.to_list() == [
        [0, 1, None, None, None],
        [5, None, 7, 8, 9],
        [None, None, None],
    ]
    array = ak.from_buffers(
        form, length, container, highlevel=False, allow_noncanonical_form=True
    )
    assert isinstance(array.content, ak.contents.BitMaskedArray)
    assert array.content._length is unknown_length
    assert array.content._length_generator is not None
    assert array.content.length == 13
    assert array.content._length == 13
    assert array.content._length_generator is None
    assert array.to_list() == [
        [0, 1, None, None, None],
        [5, None, 7, 8, 9],
        [None, None, None],
    ]


def test_RecordArray():
    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=lambda: 13,
    )
    assert array._length is unknown_length
    assert array.length == 13
    assert array._length == 13
    assert array.to_list() == [{"x": i} for i in range(13)]
    simplified = array.simplified(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=lambda: 13,
    )
    assert simplified._length is unknown_length
    assert simplified.length == 13
    assert simplified._length == 13
    assert simplified.to_list() == [{"x": i} for i in range(13)]
    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=lambda: unknown_length,
    ).to_tuple()
    assert array._length is unknown_length
    assert array.length == 13
    assert array._length == 13
    assert array.to_list() == [(i,) for i in range(13)]

    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=None,
    )
    assert array._length is unknown_length
    assert array.length == 13
    assert array._length == 13
    assert array.to_list() == [{"x": i} for i in range(13)]
    simplified = array.simplified(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=None,
    )
    assert simplified._length is unknown_length
    assert simplified.length == 13
    assert simplified._length == 13
    assert simplified.to_list() == [{"x": i} for i in range(13)]
    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=lambda: unknown_length,
    ).to_tuple()
    assert array._length is unknown_length
    assert array.length == 13
    assert array._length == 13
    assert array.to_list() == [(i,) for i in range(13)]

    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=10,
        length_generator=None,
    )
    assert array._length == 10
    assert array.length == 10
    assert array._length == 10
    assert array.to_list() == [{"x": i} for i in range(10)]
    simplified = array.simplified(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=10,
        length_generator=None,
    )
    assert simplified._length == 10
    assert simplified.length == 10
    assert simplified._length == 10
    assert simplified.to_list() == [{"x": i} for i in range(10)]
    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=10,
        length_generator=lambda: 10,
    ).to_tuple()
    assert array._length == 10
    assert array.length == 10
    assert array._length == 10
    assert array.to_list() == [(i,) for i in range(10)]

    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=lambda: 10,
    )
    assert array._length is unknown_length
    assert array.length == 10
    assert array._length == 10
    assert array.to_list() == [{"x": i} for i in range(10)]
    simplified = array.simplified(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=lambda: 10,
    )
    assert simplified._length is unknown_length
    assert simplified.length == 10
    assert simplified._length == 10
    assert simplified.to_list() == [{"x": i} for i in range(10)]
    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(13))],
        ["x"],
        length=unknown_length,
        length_generator=lambda: 10,
    ).to_tuple()
    assert array._length is unknown_length
    assert array.length == 10
    assert array._length == 10
    assert array.to_list() == [(i,) for i in range(10)]

    array = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 5, 10, 11], dtype=np.int64)),
        ak.contents.RecordArray(
            [ak.contents.NumpyArray(np.arange(13))],
            ["x"],
            length=13,
            length_generator=None,
        ),
    )
    form, length, buffers = ak.to_buffers(array)
    container = {key: (lambda b=buffer: b) for key, buffer in buffers.items()}
    array = ak.from_buffers(form, length, container, highlevel=False)
    assert isinstance(array.content, ak.contents.RecordArray)
    assert array.content._length is unknown_length
    assert array.content._length_generator is not None
    assert array.content.length == 11
    assert array.content._length == 11
    assert array.content._length_generator is None
    assert array.to_list() == [
        [{"x": i} for i in range(5)],
        [{"x": i} for i in range(5, 10)],
        [{"x": i} for i in range(10, 11)],
    ]


def test_RegularArray():
    array = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(13)),
        size=5,
        zeros_length=unknown_length,
        zeros_length_generator=lambda: 2,
    )
    assert array._length == 2
    assert array.length == 2
    assert array._length == 2
    assert array.to_list() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    simplified = array.simplified(
        ak.contents.NumpyArray(np.arange(13)),
        size=5,
        zeros_length=unknown_length,
        zeros_length_generator=lambda: 2,
    )
    assert simplified._length == 2
    assert simplified.length == 2
    assert simplified._length == 2
    assert simplified.to_list() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    array = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(13)),
        size=5,
        zeros_length=unknown_length,
        zeros_length_generator=None,
    )
    assert array._length == 2
    assert array.length == 2
    assert array._length == 2
    assert array.to_list() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    simplified = array.simplified(
        ak.contents.NumpyArray(np.arange(13)),
        size=5,
        zeros_length=unknown_length,
        zeros_length_generator=None,
    )
    assert simplified._length == 2
    assert simplified.length == 2
    assert simplified._length == 2
    assert simplified.to_list() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    array = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(13)),
        size=0,
        zeros_length=unknown_length,
        zeros_length_generator=lambda: 2,
    )
    assert array._length is unknown_length
    assert array.length == 2
    assert array._length == 2
    assert array.to_list() == [[], []]
    simplified = array.simplified(
        ak.contents.NumpyArray(np.arange(13)),
        size=0,
        zeros_length=unknown_length,
        zeros_length_generator=lambda: 2,
    )
    assert simplified._length is unknown_length
    assert simplified.length == 2
    assert simplified._length == 2
    assert simplified.to_list() == [[], []]

    array = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(13)),
        size=0,
        zeros_length=unknown_length,
        zeros_length_generator=None,
    )
    assert array._length is unknown_length
    with pytest.raises(
        AssertionError,
        match="RegularArray length must be an integer for an array with concrete data",
    ):
        assert array.length == 2
    simplified = array.simplified(
        ak.contents.NumpyArray(np.arange(13)),
        size=0,
        zeros_length=unknown_length,
        zeros_length_generator=None,
    )
    assert simplified._length is unknown_length
    with pytest.raises(
        AssertionError,
        match="RegularArray length must be an integer for an array with concrete data",
    ):
        assert simplified.length == 2

    array = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 5, 10, 13], dtype=np.int64)),
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.arange(30)),
            size=2,
            zeros_length=0,
            zeros_length_generator=None,
        ),
    )
    form, length, buffers = ak.to_buffers(array)
    container = {key: (lambda b=buffer: b) for key, buffer in buffers.items()}
    array = ak.from_buffers(form, length, container, highlevel=False)
    assert isinstance(array.content, ak.contents.RegularArray)
    assert array.content._length is unknown_length
    assert array.content._zeros_length_generator is not None
    assert array.content.length == 13
    assert array.content._length == 13
    assert array.content._zeros_length_generator is None
    assert array.to_list() == [
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
        [[20, 21], [22, 23], [24, 25]],
    ]
