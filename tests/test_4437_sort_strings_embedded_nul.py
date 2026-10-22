# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("as_bytes", [True, False])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("sliced", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        ["a\x00c", "a\x00b"],
        ["a\x00c", "a\x00b", "a\x00", "a", "", "\x00z", "\x00a", "é\x00b", "é\x00a"],
        ["a\x00c", "a\x00b", "a\x00b", "\x00", "", "a\x00"] * 20,
    ],
)
def test_sort_strings_embedded_nul(values, sliced, nested, as_bytes, stable, ascending):
    if as_bytes:
        values = [value.encode() for value in values]
    groups = [values, [], list(reversed(values))] if nested else [values]
    data = groups if nested else values
    if sliced:
        sentinel = [values[0]] if nested else values[0]
        array = ak.Array([sentinel, *data, sentinel])[1:-1]
    else:
        array = ak.Array(data)

    indices = ak.argsort(array, ascending=ascending, stable=stable)
    sorted_array = ak.sort(array, ascending=ascending, stable=stable)
    expected = [sorted(group, reverse=not ascending) for group in groups]
    expected_array = expected if nested else expected[0]
    assert sorted_array.to_list() == expected_array
    assert array[indices].to_list() == expected_array

    results = indices.to_list() if nested else [indices.to_list()]
    for group, result in zip(groups, results, strict=True):
        assert sorted(result) == list(range(len(group)))
        if stable:
            assert result == sorted(
                range(len(group)), key=group.__getitem__, reverse=not ascending
            )
