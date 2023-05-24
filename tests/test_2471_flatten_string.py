# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


def test():
    array = ak.Array(
        ["is", "this", "real", "life", "or", "is", "this", "just", "fantasy"]
    )
    with pytest.raises(ValueError, match=r"strings cannot be directly flattened"):
        ak.flatten(array, axis=1)

    array = ak.Array(
        ["is", "this", "real", "life", "or", "is", "this", "just", "fantasy"]
    )
    assert ak.to_layout(array) == ak.flatten(array, axis=0, highlevel=False)

    array_ragged = ak.Array(
        [["is", "this", "real", "life"], ["or", "is", "this", "just", "fantasy"]]
    )
    assert ak.almost_equal(array, ak.flatten(array_ragged, axis=1))

    with pytest.raises(ValueError, match=r"strings cannot be directly flattened"):
        ak.flatten(array_ragged, axis=2)
