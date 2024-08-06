# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test():
    arr = ak.Array([[1, 3, 4], 5])
    tarr = arr.layout.to_typetracer()

    with pytest.raises(ak.errors.AxisError, match="exceeds the depth of this array"):
        ak.flatten(tarr)
