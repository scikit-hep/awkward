# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    assert ak.to_numpy(ak.Array([{}, {}])).tolist() == [(), ()]
