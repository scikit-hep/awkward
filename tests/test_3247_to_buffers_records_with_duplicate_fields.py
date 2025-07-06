from __future__ import annotations

import pickle

import pytest

import awkward as ak


def test_to_from_buffers():
    a = ak.Array({"a": [1, 2, 3]})[["a", "a"]]
    b = ak.from_buffers(*ak.to_buffers(a))

    assert ak.to_list(b) == ak.to_list(a)
    assert ak.type(b) == ak.type(a)


def test_pickle():
    a = ak.Array({"a": [1, 2, 3]})[["a", "a"]]
    b = pickle.loads(pickle.dumps(a))

    assert ak.to_list(b) == ak.to_list(a)
    assert ak.type(b) == ak.type(a)


def test_arrow():
    pytest.importorskip("pyarrow")

    a = ak.Array({"a": [1, 2, 3]})[["a", "a"]]
    b = ak.from_arrow(ak.to_arrow(a))

    assert ak.to_list(b) == ak.to_list(a)
    assert ak.type(b) == ak.type(a)
