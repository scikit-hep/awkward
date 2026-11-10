# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import itertools

import numpy as np
import pytest

import awkward as ak

BOUNDS = [None, -4, -2, -1, 0, 1, 2, 3]
STEPS = [None, 1, 2, -1, -2]
SLICES = [slice(*args) for args in itertools.product(BOUNDS, BOUNDS, STEPS)]


def layouts():
    flat = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64))
    return {
        "numpy": flat,
        "list_offset": ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 1, 3, 3])), flat
        ),
        "record": ak.contents.RecordArray(
            [flat, ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0]))], ["x", "y"]
        ),
        "regular": ak.contents.RegularArray(flat, 1),
    }


@pytest.mark.parametrize("kind", ["numpy", "list_offset", "record", "regular"])
@pytest.mark.parametrize("where", SLICES, ids=str)
def test_outer_slice_matches_python(kind, where):
    array = ak.Array(layouts()[kind])
    expected = array.to_list()[where]
    result = array[where]
    assert result.to_list() == expected
    assert len(result) == len(expected)


@pytest.mark.parametrize("where", SLICES, ids=str)
def test_multidimensional_slice_matches_numpy(where):
    array = ak.Array(np.arange(12).reshape(3, 4))
    expected = np.arange(12).reshape(3, 4)
    assert array[where, :].to_list() == expected[where, :].tolist()
    assert array[:, where].to_list() == expected[:, where].tolist()
    assert array[where].to_list() == expected[where].tolist()
