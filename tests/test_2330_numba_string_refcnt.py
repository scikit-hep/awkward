# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import gc
import os

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")
psutil = pytest.importorskip("psutil")

ak.numba.register_and_check()


def test_lots_of_strings():
    process = psutil.Process(os.getpid())

    @numba.njit
    def f(arr):
        result = 0
        for i in range(len(arr)):
            result += (arr[i] == "mov")
        return result

    arr = ak.Array(["aaa" * 1024**2])
    f(arr)  # compile it

    start_memory = process.memory_info().rss

    for i in range(10):
        f(arr)
        gc.collect()

    assert (process.memory_info().rss - start_memory) / 31457280.0 < 0.9
