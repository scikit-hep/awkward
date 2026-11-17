# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("typetracer", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        ["ab", "c"],
        [[1, 2], [3]],
        [([1, 2], [3]), ([], [4, 5])],
        [([1, 2], 3), ([], 4)],
    ],
)
def test_stack_requires_a_tuple_at_axis(data, typetracer):
    array = ak.Array(data)
    if typetracer:
        array = ak.Array(array.layout.to_typetracer(forget_length=True))
    with pytest.raises(ak.errors.AxisError, match="no tuple records at axis=-1"):
        ak.stack(array)


def test_stack_selects_outer_tuple_at_shared_depth():
    array = ak.Array([((1, 2), (3, 4)), ((5, 6), (7, 8))])
    expected = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
    assert ak.stack(array).to_list() == expected
    assert ak.stack(array, axis=0).to_list() == expected
    with pytest.raises(ak.errors.AxisError, match="axis=1 exceeds the depth"):
        ak.stack(array, axis=1)
