# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward._nplikes.cupy as ak_cupy
from awkward._nplikes.array_module import _nplike_repeat_has_array_repeats
from awkward._nplikes.cupy import Cupy

nplike = Cupy.instance()

COUNTS = [[2, 1, 0], [2, 1, 0, 0], [1, 0, 2], [0, 2, 0], [0, 0, 0], [3], [0], []]


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp.cuda.Device().synchronize()
    cp._default_memory_pool.free_all_blocks()


@pytest.mark.parametrize("counts", COUNTS)
@pytest.mark.parametrize("fallback", [False, True])
def test_repeat_with_array_repeats(counts, fallback, monkeypatch):
    if fallback:
        # the hand-rolled path, still taken on CuPy < 14.1
        monkeypatch.setattr(
            ak_cupy, "_nplike_repeat_has_array_repeats", lambda module: False
        )
    x = np.arange(len(counts), dtype=np.int64) * 10
    out = nplike.repeat(cp.asarray(x), cp.asarray(np.array(counts, dtype=np.int64)))
    assert cp.asnumpy(out).tolist() == np.repeat(x, counts).tolist()


class FakeModule:
    int64 = np.int64
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)

    def __init__(self, error):
        self._error = error

    def repeat(self, x, repeats):
        raise self._error("repeats must be an integer")


def test_probe_detects_array_repeats():
    assert _nplike_repeat_has_array_repeats(np)


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_probe_detects_missing_array_repeats(error):
    # a fresh module per case, so the cached result of another case is not reused
    assert not _nplike_repeat_has_array_repeats(FakeModule(error))
