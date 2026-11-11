# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

SHAPES = [(2, 1, 0), (0, 2, 0), (1, 1, 1, 0), (2, 0, 3)]


@pytest.mark.parametrize("shape", SHAPES)
def test_masked(shape):
    array = np.ma.MaskedArray(np.zeros(shape), mask=np.zeros(shape, bool))
    result = ak.from_numpy(array)
    assert result.to_list() == np.zeros(shape).tolist()
    assert result.type.content.size == shape[1]


@pytest.mark.parametrize("shape", SHAPES)
def test_regulararray(shape):
    result = ak.from_numpy(np.zeros(shape), regulararray=True)
    assert result.to_list() == np.zeros(shape).tolist()
    assert result.type.content.size == shape[1]
