# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from types import MappingProxyType

import pytest

import awkward as ak


@pytest.mark.parametrize(
    "factory", [list, dict, tuple, lambda: iter(()), lambda: MappingProxyType({})]
)
@pytest.mark.parametrize("highlevel", [False, True])
def test_no_arrays(factory, highlevel):
    with pytest.raises(ValueError, match=r"ak\.zip requires at least one array"):
        ak.zip(factory(), highlevel=highlevel)


@pytest.mark.parametrize("arrays", [[[]], {"x": []}, [[], []], {"x": [], "y": []}])
def test_empty_arrays_are_valid(arrays):
    result = ak.zip(arrays)
    assert ak.to_list(result) == []
    assert len(result.fields) == len(arrays)
