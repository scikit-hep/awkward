# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32])
def test_no_builds_after_warmup_when_base_pointer_and_length_change(dtype, monkeypatch):
    from cuda.compute import _cccl_interop, _jit

    from awkward._connect.cuda import _indexed_fill

    _indexed_fill._setup.cache_clear()

    def run(length, base):
        values = [-1, 0, 2, -3, 5] if dtype != np.uint32 else [0, 1, 2, 3, 5]
        host = np.resize(np.asarray(values, dtype=dtype), max(length, 1))
        source = cp.asarray(host)
        output = cp.full(length + 7, 123, dtype=cp.int64)
        _indexed_fill.fill(output, 3, source, length, base)
        expected = np.full(length + 7, 123, dtype=np.int64)
        x = host[:length].astype(np.int64)
        expected[3 : 3 + length] = np.where(x < 0, -1, x + int(base))
        np.testing.assert_array_equal(cp.asnumpy(output), expected)

    run(64, 10)

    def unexpected(*args, **kwargs):
        pytest.fail("Warmed IndexedArray_fill rebuilt for runtime-only changes")

    monkeypatch.setattr(_cccl_interop, "call_build", unexpected)
    monkeypatch.setattr(_jit._StatefulOp, "compile", unexpected)
    monkeypatch.setattr(_indexed_fill, "make_unary_transform", unexpected)
    for length, base in [
        (64, 11),
        (32, np.int64(12)),
        (129, -3),
        (1, 0),
        (0, 20),
        (64, 10),
    ]:
        run(length, base)


def test_concurrent_streams_keep_separate_runtime_bases():
    from awkward._connect.cuda import _indexed_fill

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
    results = []
    for repeat in range(3):
        for i, stream in enumerate(streams):
            with stream:
                source = cp.arange(4096, dtype=cp.int64)
                output = cp.empty_like(source)
                base = 100 * repeat + i
                _indexed_fill.fill(output, 0, source, source.size, base)
                results.append((stream, output, base))
    for stream, output, base in results:
        stream.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(output), np.arange(4096) + base)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32])
def test_public_kernel_dispatch_matches_cpu(dtype):
    from awkward._backends.cupy import CupyBackend
    from awkward._backends.numpy import NumpyBackend

    key = ("awkward_IndexedArray_fill", np.int64, dtype)
    values = [-1, 0, 2, -3, 5] if dtype != np.uint32 else [0, 1, 2, 3, 5]
    for base in (-10, 0, 11, 20):
        source = np.asarray(values, dtype=dtype)
        reference = np.full(11, 123, dtype=np.int64)
        NumpyBackend.instance()[key](reference, 3, source, len(source), base)
        output = cp.full(11, 123, dtype=cp.int64)
        CupyBackend.instance()[key](output, 3, cp.asarray(source), len(source), base)
        np.testing.assert_array_equal(cp.asnumpy(output), reference)
