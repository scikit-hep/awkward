# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


def test_simple(tmp_path):
    filename = os.path.join(tmp_path, "whatever.feather")

    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    ak.to_feather(array, filename)

    array2 = ak.from_feather(filename)

    assert array2.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_complex(tmp_path):
    filename = os.path.join(tmp_path, "whatever.feather")

    array = ak.Array(
        [
            [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
            [],
            [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
        ]
    )

    ak.to_feather(array, filename)

    array2 = ak.from_feather(filename)

    assert array2.tolist() == [
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}],
        [],
        [{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}],
    ]
