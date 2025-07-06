from __future__ import annotations

import awkward as ak


def test():
    assert repr(ak.Array([])) == "<Array [] type='0 * unknown'>"
