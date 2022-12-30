# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak


def test():
    array = ak.Array([[3.14]])
    first_slice = ak.Array([True, None])[:1]  # type is 1 * ?bool
    second_slice = 0
    with pytest.raises(ValueError):
        array[first_slice, second_slice]
