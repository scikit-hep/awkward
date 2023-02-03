# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    x = ak.with_parameter([1, 2, np.datetime64("now")], "x", 100).layout
    y = ak.contents.NumpyArray(np.array([99, 88, 77], dtype=np.int64))
    result = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 0, 1], dtype=np.int64)),
        [x, y],
        parameters={"x": 100},
    )
    assert result.parameters == {"x": 100}

    result2 = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 1])),
        ak.index.Index64(np.array([0, 1, 0, 1])),
        [x, y],
        parameters={"x": 200},
    )
    assert result2.parameters == {"x": 200}

    result3 = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 1, 1])),
        ak.index.Index64(np.array([0, 1, 0, 1])),
        [x, y],
    )
    assert result3.parameters == {"x": 100}
