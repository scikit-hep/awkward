# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    first = ak.operations.from_numpy(np.array([1, 2, 3]))
    deltas = ak.highlevel.Array([[1, 2], [1, 2], [1, 2, 3]])
    assert np.hstack(
        (
            first[:, np.newaxis],
            ak.operations.fill_none(ak.operations.pad_none(deltas, 3, axis=-1), 999),
        )
    ).tolist() == [[1, 1, 2, 999], [2, 1, 2, 999], [3, 1, 2, 3]]
