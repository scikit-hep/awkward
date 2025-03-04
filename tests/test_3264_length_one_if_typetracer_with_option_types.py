# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import awkward as ak


def test():
    arr = ak.Array([[1], [2, 3], [1, 2, 4, 5]])[[0, None, 2]]
    l1 = ak.typetracer.length_one_if_typetracer(ak.to_backend(arr, "typetracer"))

    assert l1.to_list() == [None]
    assert str(l1.type) == "1 * option[var * int64]"
