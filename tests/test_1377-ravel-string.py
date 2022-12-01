# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_ListOffsetArray():
    layout = ak.from_iter(
        ["hello", "wo", "rld!", "what's", "occurring"], highlevel=False
    )
    assert isinstance(layout, ak.contents.ListOffsetArray)
    flattened = ak.ravel(layout, highlevel=False)
    assert isinstance(flattened, ak.contents.Content)
    assert flattened.to_list() == ["hello", "wo", "rld!", "what's", "occurring"]


def test_ListArray():
    layout = ak.from_iter(
        ["hello", "wo", "rld!", "what's", "occurring"], highlevel=False
    )
    assert isinstance(layout, ak.contents.ListOffsetArray)
    as_list = ak.contents.ListArray(
        layout.starts, layout.stops, layout.content, parameters=layout.parameters
    )
    assert isinstance(as_list, ak.contents.ListArray)
    flattened = ak.ravel(as_list, highlevel=False)
    assert isinstance(flattened, ak.contents.Content)
    assert flattened.to_list() == ["hello", "wo", "rld!", "what's", "occurring"]


def test_RegularArray():
    layout = ak.from_iter(["jack", "back", "tack", "rack", "sack"], highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    as_regular = layout.to_RegularArray()
    assert isinstance(as_regular, ak.contents.RegularArray)
    flattened = ak.ravel(as_regular, highlevel=False)
    assert isinstance(flattened, ak.contents.Content)
    assert flattened.to_list() == ["jack", "back", "tack", "rack", "sack"]
