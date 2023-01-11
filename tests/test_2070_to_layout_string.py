# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


@pytest.mark.parametrize("string", ["hello world!", b"hello world!"])
def test(string):
    with pytest.raises(TypeError):
        ak.to_layout(string, allow_other=False)

    assert ak.to_layout(string, allow_other=True) == string


@pytest.mark.parametrize("string", ["hello world!", b"hello world!"])
def test_ufunc(string):
    array = ak.Array([string, string])
    assert ak.all(array == string)
