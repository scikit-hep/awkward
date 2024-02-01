# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    array = ak.Array([[True, False, True], [False, False, True, False], [True, False]])

    assert ak.argmin(array, axis=1).tolist() == [1, 0, 1]
    assert ak.argmax(array, axis=1).tolist() == [0, 2, 0]
