# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize(
    "distincts, length, groups",
    [
        ([], 0, 0),
        ([], 0, 3),
        ([0, -1, 2, -1, -1, 5], 6, 2),
        ([-1] * 6, 6, 2),
        ([0, 0, -1, 1, -1, -1], 6, 2),
        ([-2, -1, -1, -1, 0, -1], 6, 2),
        ([0, -1, 2, -1, -1, 5, 99, 99], 6, 2),
        ([i if i % 3 else -1 for i in range(513)], 513, 1),
        ([i if i % 3 else -1 for i in range(1026)], 1026, 3),
        ([0, -1, 2] * 600, 1800, 600),
    ],
)
def test_cpu_parity(distincts, length, groups):
    from awkward._backends.cupy import CupyBackend
    from awkward._backends.numpy import NumpyBackend

    source = np.asarray(distincts, dtype=np.int64)
    # Check unused output capacity as well as every canonical output element.
    starts = np.full(groups + 2, -99, dtype=np.int64)
    stops = starts.copy()
    gpu_starts, gpu_stops = map(cp.asarray, (starts, stops))
    # A nonzero base offset exercises a device view rather than only allocations.
    gpu_source = cp.asarray(np.concatenate(([123], source)))[1:]
    key = ("awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",) + (
        np.int64,
    ) * 3
    error = NumpyBackend.instance()[key](starts, stops, source, length, groups)
    assert not error.str
    CupyBackend.instance()[key](gpu_starts, gpu_stops, gpu_source, length, groups)
    ak._connect.cuda.synchronize_cuda()
    np.testing.assert_array_equal(cp.asnumpy(gpu_starts), starts)
    np.testing.assert_array_equal(cp.asnumpy(gpu_stops), stops)


def test_no_explicit_host_scalar_conversion(monkeypatch):
    from awkward._connect.cuda import _compute

    operation = _compute.awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64

    source = cp.asarray([0, -1, 2, -1, -1, 5], dtype=cp.int64)
    starts, stops = cp.empty(2, dtype=cp.int64), cp.empty(2, dtype=cp.int64)

    def reject(*args, **kwargs):
        raise AssertionError("kernel attempted an explicit device-to-host transfer")

    monkeypatch.setattr(cp, "asnumpy", reject)
    monkeypatch.setattr(_compute, "int", reject, raising=False)
    operation(starts, stops, source, 6, 2)
    cp.cuda.Device().synchronize()
