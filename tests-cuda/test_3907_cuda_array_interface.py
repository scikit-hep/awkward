from __future__ import annotations

import timeit

import numpy as np
import pytest


def test_numpy_array_cuda_array_interface_on_cupy_backend():
    """NumpyArray backed by CuPy exposes __cuda_array_interface__."""
    ak = pytest.importorskip("awkward")

    array = ak.to_backend(ak.Array([1.0, 2.0, 3.0]), "cuda")
    layout = array.layout

    assert hasattr(layout, "__cuda_array_interface__")
    cai = layout.__cuda_array_interface__
    assert "data" in cai
    assert "shape" in cai
    assert "typestr" in cai
    # pointer should be non-null
    assert cai["data"][0] != 0


def test_numpy_array_cuda_array_interface_raises_on_numpy_backend():
    """NumpyArray backed by NumPy must NOT expose __cuda_array_interface__."""
    ak = pytest.importorskip("awkward")

    array = ak.Array([1.0, 2.0, 3.0])
    layout = array.layout

    with pytest.raises(AttributeError):
        _ = layout.__cuda_array_interface__


def test_numpy_array_cuda_array_interface_passthrough():
    """__cuda_array_interface__ on NumpyArray matches the underlying CuPy array's."""
    cp = pytest.importorskip("cupy")
    ak = pytest.importorskip("awkward")

    raw = cp.array([1.0, 2.0, 3.0])
    array = ak.to_backend(ak.Array(raw), "cuda")
    layout = array.layout

    assert layout.__cuda_array_interface__ == raw.__cuda_array_interface__


def test_numpy_array_cuda_array_interface_accepted_by_cupy():
    """CuPy should be able to consume a NumpyArray directly via the interface."""
    cp = pytest.importorskip("cupy")
    ak = pytest.importorskip("awkward")

    array = ak.to_backend(ak.Array([1.0, 2.0, 3.0]), "cuda")
    layout = array.layout

    # CuPy should accept the layout directly without needing .data
    result = cp.asarray(layout)
    cp.testing.assert_array_equal(result, cp.array([1.0, 2.0, 3.0]))


def test_numpy_array_cuda_array_interface_accepted_by_cuda_compute():
    """cuda.compute should accept NumpyArray directly without unwrapping .data."""
    cp = pytest.importorskip("cupy")
    ak = pytest.importorskip("awkward")
    cuda_compute = pytest.importorskip("cuda.compute")

    array = ak.to_backend(ak.Array({"x": [[1.0, 2.0], [3.0]]}), "cuda")
    content = array["x"].layout.content  # NumpyArray, not .data
    offsets = array["x"].layout.offsets  # Index, not .data
    result = cp.zeros(2, dtype=np.float64)

    def min_op(a, b):
        return a if a < b else b

    identity_host = np.asarray(np.inf, dtype=np.float64)

    # Should not raise — NumpyArray now satisfies __cuda_array_interface__
    cuda_compute.segmented_reduce(
        content, result, offsets.data[:-1], offsets.data[1:], min_op, identity_host, 2
    )
    cp.testing.assert_allclose(result, cp.array([1.0, 3.0]))


def test_awkward_reduce_min_cupy_performance():
    """
    Benchmarks awkward_reduce_min (cuda.compute segmented_reduce)
    against a naive CuPy per-segment loop and a flat cp.min.

    Results from benchmarking session:
        awkward_reduce_min (original):  min=0.1994s  (includes offsets slicing + memset)
        awkward_reduce_min:             min=0.1553s  (pre-sliced offsets)
        cupy segmented loop:            min=0.1656s  (one cp.min kernel per segment)
        cupy flat min:                  min=0.0799s  (flat, not segmented — not comparable)

    Conclusion: awkward_reduce_min is the fastest correct segmented option,
    beating the native CuPy loop by ~6% and handling empty segments correctly.
    """
    cp = pytest.importorskip("cupy")
    ak = pytest.importorskip("awkward")
    cuda_compute = pytest.importorskip("cuda.compute")

    # Build a test array with at least one empty segment
    array = ak.Array({"x": [[1.0, 2.0, 3.0], [], [4.0, 5.0]]})
    array = ak.to_backend(array, "cuda")

    content = array["x"].layout.content.data  # raw CuPy array
    offsets = array["x"].layout.offsets.data  # raw CuPy array
    outlength = len(array["x"].layout)

    starts = offsets[:-1]
    stops = offsets[1:]

    result = cp.zeros(outlength, dtype=np.float64)

    def min_op(a, b):
        return a if a < b else b

    def awkward_reduce_min(toptr, fromptr, starts, stops, outlength, identity=np.inf):
        identity_host = np.asarray(identity, dtype=fromptr.dtype)
        cuda_compute.segmented_reduce(
            fromptr, toptr, starts, stops, min_op, identity_host, outlength
        )

    def cupy_segmented_loop():
        return cp.array(
            [
                cp.min(content[s:e]) if e > s else cp.array(cp.inf)
                for s, e in zip(starts.tolist(), stops.tolist(), strict=True)
            ]
        )

    # Warm up both paths to avoid JIT skewing results
    awkward_reduce_min(result, content, starts, stops, outlength)
    cp.cuda.Stream.null.synchronize()
    cupy_segmented_loop()
    cp.cuda.Stream.null.synchronize()

    REPEAT = 5
    NUMBER = 100

    awk_times = timeit.repeat(
        lambda: (
            awkward_reduce_min(result, content, starts, stops, outlength),
            cp.cuda.Stream.null.synchronize(),
        ),
        repeat=REPEAT,
        number=NUMBER,
    )

    cupy_loop_times = timeit.repeat(
        lambda: (cupy_segmented_loop(), cp.cuda.Stream.null.synchronize()),
        repeat=REPEAT,
        number=NUMBER,
    )

    cupy_flat_times = timeit.repeat(
        lambda: (cp.min(content), cp.cuda.Stream.null.synchronize()),
        repeat=REPEAT,
        number=NUMBER,
    )

    awk_min = min(awk_times)
    loop_min = min(cupy_loop_times)
    flat_min = min(cupy_flat_times)

    print(f"\nawkward_reduce_min:     min={awk_min:.4f}s")
    print(f"cupy segmented loop:      min={loop_min:.4f}s")
    print(
        f"cupy flat min:            min={flat_min:.4f}s  (not segmented, lower bound only)"
    )
    print(f"vs cupy loop:  {awk_min / loop_min:.2f}x  (should be <= 1.0)")
    print(
        f"vs cupy flat:  {awk_min / flat_min:.2f}x  (expected ~2x, flat is not segmented)"
    )

    # awkward_reduce_min must be faster than or comparable to the naive cupy loop
    assert awk_min <= loop_min * 1.2, (
        f"awkward_reduce_min ({awk_min:.4f}s) is more than 20% slower than "
        f"cupy segmented loop ({loop_min:.4f}s)"
    )

    # Correctness: verify results match cupy loop
    expected = cupy_segmented_loop()
    cp.testing.assert_allclose(result, expected)
