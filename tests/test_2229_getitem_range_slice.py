# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length


def test_known_scalar_regularization():
    layout = ak.to_layout([1, 2, 3, 4, 5, 6]).to_typetracer(forget_length=False)
    assert layout[:].length == 6
    assert layout[:3].length == 3
    assert layout[2:4].length == 2
    assert layout[::-1].length == 6


def test_unknown_length_regularization():
    layout = ak.to_layout([1, 2, 3, 4, 5, 6]).to_typetracer(forget_length=False)
    assert layout[unknown_length::].length == unknown_length
    assert layout[:unknown_length:].length == unknown_length
    assert layout[::unknown_length].length == unknown_length


def test_unknown_scalar_regularization():
    layout = ak.to_layout([1, 2, 3, 4, 5, 6]).to_typetracer(forget_length=False)
    assert layout[layout[0] : :].length == unknown_length
    assert layout[: layout[0]].length == unknown_length
    assert layout[:: layout[0]].length == unknown_length


def test_slice_regularization_unknown_length():
    layout = ak.to_layout([1, 2, 3, 4, 5, 6]).to_typetracer(forget_length=True)
    assert layout[:].length == unknown_length
    assert layout[:3].length == unknown_length
    assert layout[2:4].length == unknown_length
    assert layout[::-1].length == unknown_length
    assert layout[unknown_length::].length == unknown_length
    assert layout[:unknown_length:].length == unknown_length
    assert layout[::unknown_length].length == unknown_length
    assert layout[layout[0] : :].length == unknown_length
    assert layout[: layout[0]].length == unknown_length
    assert layout[:: layout[0]].length == unknown_length
