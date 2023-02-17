# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np

import awkward as ak


def test():
    index = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 5]),
            ak.contents.NumpyArray(
                np.array([True, False, False, True, True, False, False], dtype=np.bool_)
            ),
        )
    )
    array = ak.Array([[0, 1, 2], [3, 4]])
    result = array[index]
    assert result.tolist() == [[0], [3, 4]]
