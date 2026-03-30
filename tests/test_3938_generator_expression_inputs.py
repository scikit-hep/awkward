# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak

to_list = ak.operations.to_list


def test_concatenate():
    result = ak.concatenate([i] for i in range(3))
    assert to_list(result) == [0, 1, 2]


def test_zip():
    result = ak.zip([i, i + 1] for i in range(2))
    assert to_list(result) == [(0, 1), (1, 2)]


def test_zip_no_broadcast():
    result = ak.zip_no_broadcast([i, i + 1] for i in range(2))
    assert to_list(result) == [(0, 1), (1, 2)]


def test_cartesian():
    result = ak.cartesian([[i, i + 1]] for i in range(2))
    assert to_list(result) == [[(0, 1), (0, 2), (1, 1), (1, 2)]]


def test_argcartesian():
    result = ak.argcartesian([[i, i + 1]] for i in range(2))
    assert to_list(result) == [[(0, 0), (0, 1), (1, 0), (1, 1)]]
