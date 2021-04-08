# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_empty_arrays_cartesian():
    one = ak.Array([])
    two = one = ak.Array([])

    with pytest.raises(ValueError) as err:
        ak.to_list(ak.cartesian([one, two]))

    assert isinstance(err.value, ValueError)

    ak.to_list(ak.concatenate([one, two], axis=0))


def test_cartesian():
    muon = ak.Array([[{"pt": 1.0}], []], with_name="muon")
    electron = ak.Array([[], [{"pt": 1.0}]], with_name="electron")

    muon = muon[muon.pt > 5]
    electron = electron[electron.pt > 5]

    leptons = ak.concatenate([muon, electron], axis=1)
    candidate = ak.firsts(leptons)
    assert ak.to_list(ak.Array(candidate)) == [None, None]

    result = ak.cartesian([candidate, candidate], axis=0)
    assert ak.to_list(result) == [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]

    result = ak.cartesian([candidate, ak.Array([[1, 2, 3], []])], axis=1)
    assert ak.to_list(result) == [None, None]

    one, two = ak.broadcast_arrays(candidate, ak.Array([[1, 2, 3], []]))
    assert ak.to_list(one) == [None, None]
    assert ak.to_list(two) == [None, None]
