from __future__ import annotations

import awkward as ak


def test():
    length0 = ak.Array([])
    assert repr(length0) == "<Array [] type='0 * unknown'>"
    assert length0._repr_mimebundle_() == {
        "text/html": "<pre>\nbackend: cpu\nnbytes: 0 B\ntype: 0 * unknown</pre>",
        "text/plain": "<Array [] type='0 * unknown'>",
    }
