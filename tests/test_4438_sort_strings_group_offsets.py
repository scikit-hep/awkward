# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("as_bytes", [True, False])
@pytest.mark.parametrize("sliced", [True, False])
@pytest.mark.parametrize(
    "case", ["flat", "jagged", "regular", "optional", "empty", "empty_groups"]
)
def test_sort_strings_group_offsets(case, sliced, as_bytes, stable, ascending):
    values = ["b", "a\x00c", "", "a\x00b", "b", "a", "é"]
    if as_bytes:
        values = [value.encode() for value in values]
    groups = [[], values, [], list(reversed(values)), []]
    if case == "flat":
        array = ak.Array(values)
    elif case == "jagged":
        array = ak.Array(groups)
    elif case == "regular":
        groups = [values, list(reversed(values))]
        array = ak.to_regular(ak.Array(groups), axis=1)
    elif case == "optional":
        groups = [None, values, [], None, list(reversed(values)), None]
        array = ak.Array(groups)
    elif case == "empty":
        array = ak.Array(values)[:0]
    else:
        groups = [[], [], []]
        array = ak.unflatten(ak.Array(values)[:0], [0, 0, 0])
    if sliced:
        # Concatenation followed by slicing leaves nonzero outer/byte offsets.
        array = ak.concatenate([array, array, array])[len(array) : 2 * len(array)]

    flat = case in ("flat", "empty")
    input_groups = [array.to_list()] if flat else array.to_list()
    expected_groups = [
        None if group is None else sorted(group, reverse=not ascending)
        for group in input_groups
    ]
    expected = expected_groups[0] if flat else expected_groups
    result = ak.sort(array, ascending=ascending, stable=stable)
    carry = ak.argsort(array, ascending=ascending, stable=stable)
    assert result.to_list() == expected
    assert array[carry].to_list() == expected

    carry_groups = [carry.to_list()] if flat else carry.to_list()
    for group, indices in zip(input_groups, carry_groups, strict=True):
        if group is None:
            assert indices is None
        else:
            assert sorted(indices) == list(range(len(group)))
            if stable:
                assert indices == sorted(
                    range(len(group)), key=group.__getitem__, reverse=not ascending
                )

    for forget_length in [False, True]:
        typetracer = ak.Array(array.layout.to_typetracer(forget_length=forget_length))
        assert (
            ak.sort(typetracer, ascending=ascending, stable=stable).layout.form
            == result.layout.form
        )
        assert (
            ak.argsort(typetracer, ascending=ascending, stable=stable).layout.form
            == carry.layout.form
        )
