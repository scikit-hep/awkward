# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.virtual import VirtualNDArray


@pytest.mark.parametrize("offsets_length", [5, unknown_length])
@pytest.mark.parametrize("content_length", [9, unknown_length])
def test(offsets_length, content_length):
    offset_generator = lambda: np.array([0, 2, 4, 5, 6], dtype=np.int64)  # noqa: E731
    data_generator = lambda: np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)  # noqa: E731

    offsets = ak.index.Index(
        VirtualNDArray(
            Numpy.instance(),
            shape=(offsets_length,),
            dtype=np.int64,
            generator=offset_generator,
        )
    )
    data = ak.contents.NumpyArray(
        VirtualNDArray(
            Numpy.instance(),
            shape=(content_length,),
            dtype=np.int64,
            generator=data_generator,
        )
    )
    array = ak.Array(ak.contents.ListOffsetArray(offsets, data))
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]

    buffers = {"node0-offsets": offset_generator, "node1-data": data_generator}
    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("int64", form_key="node1"), form_key="node0"
    )
    array = ak.from_buffers(form, 4, buffers)
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]

    offsets = VirtualNDArray(
        Numpy.instance(),
        shape=(offsets_length,),
        dtype=np.int64,
        generator=offset_generator,
    )
    data = VirtualNDArray(
        Numpy.instance(),
        shape=(content_length,),
        dtype=np.int64,
        generator=data_generator,
    )
    buffers = {"node0-offsets": offsets, "node1-data": data}
    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("int64", form_key="node1"), form_key="node0"
    )
    array = ak.from_buffers(form, 4, buffers)
    assert array.to_list() == [[1, 2], [3, 4], [5], [6]]
