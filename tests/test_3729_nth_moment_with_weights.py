# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    data = ak.Array([1.0, 2.0])
    weight = ak.Array([2.0, 3.0])

    # Expected second raw moment: (2*1.0**2 + 3*2.0**2) / (2 + 3) = 14 / 5 = 2.8
    expected = (2 * 1.0**2 + 3 * 2.0**2) / (2 + 3)
    observed = ak.moment(data, 2, weight=weight, axis=None)
    assert expected == observed
