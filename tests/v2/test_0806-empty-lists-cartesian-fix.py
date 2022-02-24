# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.convert.to_list


def test_empty_arrays_cartesian():
    one = ak._v2.Array([])
    two = one = ak._v2.Array([])

    with pytest.raises(ValueError) as err:
        to_list(ak._v2.operations.structure.cartesian([one, two]))

    assert isinstance(err.value, ValueError)

    to_list(ak._v2.operations.structure.concatenate([one, two], axis=0))


@pytest.mark.skip(reason="ak._v2.operations.structure.firsts to be implemented")
def test_cartesian():
    muon = ak._v2.Array([[{"pt": 1.0}], []], with_name="muon")
    electron = ak._v2.Array([[], [{"pt": 1.0}]], with_name="electron")

    muon = muon[muon.pt > 5]
    electron = electron[electron.pt > 5]

    leptons = ak._v2.operations.structure.concatenate([muon, electron], axis=1)
    candidate = ak._v2.operations.structure.firsts(leptons)
    assert to_list(ak._v2.Array(candidate)) == [None, None]

    result = ak._v2.operations.structure.cartesian([candidate, candidate], axis=0)
    assert to_list(result) == [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ]

    result = ak._v2.operations.structure.cartesian(
        [candidate, ak._v2.Array([[1, 2, 3], []])], axis=1
    )
    assert to_list(result) == [None, None]

    one, two = ak.broadcast_arrays(candidate, ak._v2.Array([[1, 2, 3], []]))
    assert to_list(one) == [None, None]
    assert to_list(two) == [None, None]
