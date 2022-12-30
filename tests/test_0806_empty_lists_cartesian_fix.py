# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_empty_arrays_cartesian():
    one = ak.Array([])
    two = one = ak.Array([])

    with pytest.raises(ValueError) as err:
        to_list(ak.operations.cartesian([one, two]))

    assert isinstance(err.value, ValueError)

    to_list(ak.operations.concatenate([one, two], axis=0))


def test_cartesian():
    muon = ak.Array([[{"pt": 1.0}], []], with_name="muon")
    electron = ak.Array([[], [{"pt": 1.0}]], with_name="electron")

    muon = muon[muon.pt > 5]
    electron = electron[electron.pt > 5]

    leptons = ak.operations.concatenate([muon, electron], axis=1)
    candidate = ak.operations.firsts(leptons)
    assert to_list(ak.Array(candidate)) == [None, None]

    result = ak.operations.cartesian([candidate, candidate], axis=0)
    assert to_list(result) == [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]

    result = ak.operations.cartesian([candidate, ak.Array([[1, 2, 3], []])], axis=1)

    assert to_list(result) == [None, None]

    one, two = ak.operations.broadcast_arrays(candidate, ak.Array([[1, 2, 3], []]))
    assert to_list(one) == [None, None]
    assert to_list(two) == [None, None]
