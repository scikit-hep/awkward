# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    record = ak.Record({"0": {"0": 0}, "1": 0})
    assert record.to_list() == {"0": {"0": 0}, "1": 0}
