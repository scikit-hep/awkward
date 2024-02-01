# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak

# 6.6 and 7.7 are inaccessible
layout = ak.contents.listoffsetarray.ListOffsetArray(
    ak.index.Index(np.array([1, 4, 4, 6], dtype=np.int64)),
    ak.contents.numpyarray.NumpyArray(
        np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7], dtype=np.float64)
    ),
)


def test_recursive():
    form, length, container = ak.to_buffers(layout)
    assert form.expected_from_buffers(recursive=True) == {
        "node0-offsets": np.dtype("int64"),
        "node1-data": np.dtype("float64"),
    }


def test_non_recursive():
    form, length, container = ak.to_buffers(layout)
    assert form.expected_from_buffers(recursive=False) == {
        "node0-offsets": np.dtype("int64")
    }
