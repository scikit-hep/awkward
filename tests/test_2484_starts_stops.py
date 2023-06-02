# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_listoffsetarray():
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64([0]), ak.contents.NumpyArray([0, 1, 2, 3])
    )
    assert layout.starts.data.tolist() == [0]
    assert layout.stops.data.tolist() == [0]

    layout = ak.contents.ListOffsetArray(
        ak.index.Index64([0, 2]), ak.contents.NumpyArray([0, 1, 2, 3])
    )
    assert layout.starts.data.tolist() == [0]
    assert layout.stops.data.tolist() == [2]


def test_listarray():
    layout = ak.contents.ListArray(
        ak.index.Index64([0]),
        ak.index.Index64([0]),
        ak.contents.NumpyArray([0, 1, 2, 3]),
    )
    assert layout.starts.data.tolist() == [0]
    assert layout.stops.data.tolist() == [0]

    layout = ak.contents.ListArray(
        ak.index.Index64([0]),
        ak.index.Index64([2]),
        ak.contents.NumpyArray([0, 1, 2, 3]),
    )
    assert layout.starts.data.tolist() == [0]
    assert layout.stops.data.tolist() == [2]


def test_regulararray():
    layout = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(0, dtype=np.int64)), size=2
    )
    assert layout.starts.data.tolist() == [0]
    assert layout.stops.data.tolist() == [0]

    layout = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.arange(2, dtype=np.int64)), size=2
    )
    assert layout.starts.data.tolist() == [0]
    assert layout.stops.data.tolist() == [2]
