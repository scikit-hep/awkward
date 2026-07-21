# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import tracemalloc

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


def test_3646_arrow_null_slowdown():
    tracemalloc.start()
    try:
        arr = ak.Array(["oof" * 2000] + [None] * 50000)
        pa_arr = pyarrow.array(arr)
        assert len(pa_arr) == 50001
        assert pa_arr[0].as_py() == "oof" * 2000

        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # With the bug, allocating the intermediate list array copies "oof"*2000
    # 50,000 times, which consumes over 300 MB of Python-tracked memory.
    # With the fix, peak memory is under 20 MB.
    assert peak < 30 * 1024 * 1024
