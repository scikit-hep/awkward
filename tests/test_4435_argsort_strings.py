# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "keys",
    [
        ["a", "a"],
        ["b", "a", "aa", "b", "", "a", "aa", ""],
        ["a"] * 100,
    ],
)
def test_argsort_strings(keys, nested, stable, ascending):
    array = ak.Array([keys, [], keys] if nested else keys)
    result = ak.argsort(array, ascending=ascending, stable=stable).to_list()
    groups = [keys, [], keys] if nested else [keys]
    results = result if nested else [result]

    for group, indices in zip(groups, results, strict=True):
        assert sorted(indices) == list(range(len(group)))
        assert [group[i] for i in indices] == sorted(group, reverse=not ascending)
        if stable:
            assert indices == sorted(
                range(len(group)), key=group.__getitem__, reverse=not ascending
            )
