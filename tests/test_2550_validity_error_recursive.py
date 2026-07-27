# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("forget_length", [False, True])
def test(forget_length):
    content = ak.contents.ListOffsetArray(
        ak.index.Index64([0, 2, 4, 4]),
        ak.contents.IndexedArray(
            ak.index.Index64([0, 0, 1, 2]),
            ak.contents.NumpyArray(np.array([0, 0, 10, 20])),
            parameters={"__array__": "categorical"},
        ),
    )
    assert (
        ak.validity_error(content)
        == 'at highlevel.content ("<class \'awkward.contents.indexedarray.IndexedArray\'>"): __array__ = "categorical" requires contents to be unique'
    )
    assert (
        ak.validity_error(content.to_typetracer(forget_length))
        == 'at highlevel.content ("<class \'awkward.contents.indexedarray.IndexedArray\'>"): __array__ = "categorical" requires contents to be unique'
    )
