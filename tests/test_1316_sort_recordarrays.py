from __future__ import annotations

import numpy as np

import awkward as ak


def test_sort_record():
    a = ak.contents.NumpyArray(np.array([50, 500, 100, 1, 200, 1000]))

    record = ak.Array(ak.contents.RecordArray([a, a[::-1]], ["a", "b"]))

    assert ak.to_list(ak.sort(record)["a"]) == [1, 50, 100, 200, 500, 1000]
    assert ak.to_list(ak.sort(record)["b"]) == [1, 50, 100, 200, 500, 1000]


def test_sort_record_tuple():
    a = ak.contents.NumpyArray(np.array([50, 500, 100, 1, 200, 1000]))

    record = ak.Array(ak.contents.RecordArray([a, a[::-1]], None))

    assert ak.to_list(ak.sort(record)["0"]) == [1, 50, 100, 200, 500, 1000]
    assert ak.to_list(ak.sort(record)["1"]) == [1, 50, 100, 200, 500, 1000]
