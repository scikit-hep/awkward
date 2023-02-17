# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("string", ["hello world!", b"hello world!"])
def test(string):
    assert ak.almost_equal(
        ak.to_layout(string),
        ak.contents.NumpyArray(
            np.array(
                [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33],
                dtype=np.uint8,
            ),
            parameters={"__array__": "char" if isinstance(string, str) else "byte"},
        ),
    )


@pytest.mark.parametrize("string", ["hello world!", b"hello world!"])
def test_ufunc(string):
    array = ak.Array([string, string])
    assert ak.all(array == string)
