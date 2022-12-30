# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    x = ak.contents.IndexedArray(
        ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([9, 6, 5], dtype=np.int16)),
        parameters={"money": "doesn't buy happiness"},
    )
    y = ak.contents.IndexedArray(
        ak.index.Index64(np.array([0, 1, 2, 4, 3], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([9, 6, 5, 8, 2], dtype=np.int16)),
        parameters={"age": "number"},
    )

    # Test that we invoke the merge pathway
    z = x._reverse_merge(y)
    assert z.to_list() == [9, 6, 5, 2, 8, 9, 9, 6]
    assert z.parameters == {"money": "doesn't buy happiness", "age": "number"}
