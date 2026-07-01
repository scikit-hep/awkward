# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import random

import numpy as np

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray
from awkward.index import Index64


def _string_array(strings):
    data = "".join(strings).encode()
    offsets = np.zeros(len(strings) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s.encode()) for s in strings])
    chars = NumpyArray(
        np.frombuffer(bytearray(data), dtype=np.uint8),
        parameters={"__array__": "char"},
    )
    layout = ListOffsetArray(
        Index64(offsets), chars, parameters={"__array__": "string"}
    )
    return ak.Array(layout)


def test_sort_strings_over_255_cumulative_chars():
    # 200 distinct 4-char strings -> 800 cumulative bytes, well past the
    # uint8 (255) wraparound that previously corrupted the sort kernel.
    random.seed(1)
    strings = [f"{i:04d}" for i in range(200)]
    shuffled = strings[:]
    random.shuffle(shuffled)

    arr = _string_array(shuffled)
    result = ak.to_list(ak.sort(arr))
    assert result == sorted(shuffled)


def test_unique_strings_adjacent_duplicates_of_differing_lengths():
    # Regression for the in-place compaction overwriting the comparison
    # region: previously yielded ['aa', 'bcde', 'bcde'].
    arr = ak.Array(["aa", "aa", "bcde", "bcde"])
    result = ak.to_list(ak._do.unique(arr.layout, axis=-1))
    assert result == ["aa", "bcde"]
