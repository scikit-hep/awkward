# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def test():
    conditionals = ak.Array([True, True, True, False, False, False])
    unionarray = ak.Array([1, 2, 3, [4, 5], [], [6]])
    otherarray = ak.Array(range(100, 106))
    result = ak.where(conditionals, unionarray, otherarray)
    assert result.tolist() == [1, 2, 3, 103, 104, 105]
    assert str(result.type) == "6 * union[int64, var * int64]"

    conditionals_tt = ak.Array(conditionals.layout.to_typetracer())
    unionarray_tt = ak.Array(unionarray.layout.to_typetracer())
    otherarray_tt = ak.Array(otherarray.layout.to_typetracer())
    result_tt = ak.where(conditionals_tt, unionarray_tt, otherarray_tt)
    assert str(result_tt.type) == "6 * union[int64, var * int64]"
