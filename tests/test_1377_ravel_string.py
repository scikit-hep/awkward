# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


@pytest.mark.parametrize(
    "data",
    [
        ["hello", "wo", "rld!", "what's", "occurring"],
        [b"hello", b"wo", b"rld!", b"what's", b"occurring"],
    ],
)
def test_ListOffsetArray(data):
    layout = ak.from_iter(data, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    flattened = ak.ravel(layout, highlevel=False)
    assert isinstance(flattened, ak.contents.Content)
    assert flattened.to_list() == data


@pytest.mark.parametrize(
    "data",
    [
        ["hello", "wo", "rld!", "what's", "occurring"],
        [b"hello", b"wo", b"rld!", b"what's", b"occurring"],
    ],
)
def test_ListArray(data):
    layout = ak.from_iter(data, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    as_list = ak.contents.ListArray(
        layout.starts, layout.stops, layout.content, parameters=layout.parameters
    )
    assert isinstance(as_list, ak.contents.ListArray)
    flattened = ak.ravel(as_list, highlevel=False)
    assert isinstance(flattened, ak.contents.Content)
    assert flattened.to_list() == data


@pytest.mark.parametrize(
    "data",
    [
        ["jack", "back", "tack", "rack", "sack"],
        [b"jack", b"back", b"tack", b"rack", b"sack"],
    ],
)
def test_RegularArray(data):
    layout = ak.from_iter(data, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    as_regular = layout.to_RegularArray()
    assert isinstance(as_regular, ak.contents.RegularArray)
    flattened = ak.ravel(as_regular, highlevel=False)
    assert isinstance(flattened, ak.contents.Content)
    assert flattened.to_list() == data
