from __future__ import annotations

import awkward as ak


def test_concat_empty_records():
    a = ak.Array([{}, {}, {}])
    b = ak.Array([{}, {}])
    c = ak.concatenate([a, b], axis=0)
    assert c.to_list() == [{}, {}, {}, {}, {}]
