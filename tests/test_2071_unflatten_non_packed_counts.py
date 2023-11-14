# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_indexed_counts():
    counts = ak.contents.IndexedArray(
        ak.index.Index64(np.arange(3)),
        ak.contents.NumpyArray(np.array([3, 0, 2], dtype=np.int64)),
    )
    assert ak.almost_equal(
        ak.unflatten([1.1, 2.2, 3.3, 4.4, 5.5], counts),
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
    )


def test_indexed_layout():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.arange(5)),
        ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)),
    )
    assert ak.almost_equal(
        ak.unflatten(layout, [3, 0, 2]),
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
    )


def test_option_counts():
    assert ak.almost_equal(
        ak.unflatten([1.1, 2.2, 3.3, 4.4, 5.5], [None, 3, None, 0, 2]),
        [None, [1.1, 2.2, 3.3], None, [], [4.4, 5.5]],
    )


def test_categorical_counts():
    pytest.importorskip("pyarrow")

    assert ak.almost_equal(
        ak.unflatten([1.1, 2.2, 3.3, 4.4, 5.5], ak.str.to_categorical([3, 0, 2])),
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
    )
