# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pickle

import numpy as np

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward.typetracer import PlaceholderArray

nplike = Numpy.instance()


def test_round_trip_placeholder():
    layout = ak.contents.RecordArray(
        [
            ak.contents.NumpyArray(
                PlaceholderArray(nplike, (4,), np.dtype(np.float64))
            ),
            ak.contents.NumpyArray([1, 2, 3, 4]),
        ],
        ["x", "y"],
    )

    form, length, container = ak.to_buffers(layout)
    result = ak.from_buffers(form, length, container, highlevel=False)

    assert isinstance(result.content("x").data, PlaceholderArray)

    assert result.content("x").data.shape == layout.content("x").data.shape
    assert result.content("x").data.dtype == layout.content("x").data.dtype


def test_pickle_nplike():
    assert pickle.loads(pickle.dumps(nplike)) is nplike
