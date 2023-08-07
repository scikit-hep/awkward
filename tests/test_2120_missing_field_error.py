# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


def test():
    array = ak.Array(
        [
            {"x": 100, "y": 200},
            {"x": 300, "y": 400},
        ]
    )
    with pytest.raises(ak.errors.FieldNotFoundError):
        array["z"]
    with pytest.raises(ak.errors.FieldNotFoundError):
        array[["z", "k"]]
    with pytest.raises(ak.errors.FieldNotFoundError):
        array[0, "z"]
