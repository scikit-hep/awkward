# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak

# from awkward.operations import to_list


def test_concatenate_as_reported():
    a = ak.Array([[1]])
    a_concat_128 = ak.concatenate([a for i in range(128)], axis=1)
    assert a_concat_128.to_list() == [[1] * 128]

    a_concat_129 = ak.concatenate([a for i in range(129)], axis=1)
    assert a_concat_129.to_list() == [[1] * 129]


def test_concatenate_inner_union_simplify_one():
    a = ak.Array([[99]])
    astr = ak.Array(["a b c d".split()])
    aa = [a for i in range(129)] + [astr]

    cu = ak.concatenate(aa, axis=1)
    assert cu.to_list() == [[99] * 129 + ["a", "b", "c", "d"]]


def test_concatenate_inner_union_simplify():
    a = ak.Array([[99]])
    amulti = ak.Array([[1, 2, "a", "b"]])
    aa = [a for i in range(129)] + [amulti]

    cu = ak.concatenate(aa, axis=1)
    assert cu.to_list() == [[99] * 129 + [1, 2, "a", "b"]]
