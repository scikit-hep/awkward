# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import pytest

import awkward as ak


def test():
    arr = ak.Array([1])

    with pytest.raises(
        TypeError,
        match="'attrs' keys must be strings, got: 1",
    ):
        arr.attrs[1] = "foo"
