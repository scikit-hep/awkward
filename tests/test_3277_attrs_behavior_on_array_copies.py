# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import awkward as ak


def test():
    arr = ak.Array([1])
    arr.attrs["foo"] = "bar"

    arr2 = ak.copy(arr)
    assert arr2.attrs == arr.attrs

    arr2.attrs["foo"] = "baz"
    assert arr2.attrs != arr.attrs
