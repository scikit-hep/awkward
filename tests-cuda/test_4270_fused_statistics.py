# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("n", [-3, 0, 1, 2, 5])
def test_fused_gpu_statistics(dtype, n):
    data = np.array([[1, 2, 3], [2, 3, 4]], dtype=dtype)
    if data.dtype.kind == "c":
        data.imag = [[1, -1, 2], [-2, 1, -1]]
    cpu = ak.Array(data)
    gpu = ak.to_backend(cpu, "cuda")
    np.testing.assert_allclose(
        ak.to_numpy(ak.to_backend(ak.moment(gpu, n, axis=-1), "cpu")),
        ak.to_numpy(ak.moment(cpu, n, axis=-1)),
        rtol=3e-6,
    )
    np.testing.assert_allclose(
        ak.to_numpy(ak.to_backend(ak.var(gpu, axis=-1), "cpu")),
        ak.to_numpy(ak.var(cpu, axis=-1)),
        rtol=3e-6,
    )


@pytest.mark.parametrize("complex_input", [False, True])
def test_centered_kernel_prefix_and_empty_bins(complex_input, monkeypatch):
    from awkward._backends.cupy import CupyBackend

    dtype = cp.float64
    values = cp.asarray([99, 1, 2, 3, 88], dtype=dtype)
    means = cp.asarray([1.5, 0, 3], dtype=dtype)
    name = "awkward_reduce_centered_sumofsquares"
    expected = [0.5, 0, 0]
    if complex_input:
        values = cp.asarray([99, 1 + 1j, 2 - 1j, 3 + 2j, 88], dtype=cp.complex128).view(
            dtype
        )
        means = means.astype(cp.complex128).view(dtype)
        name += "_complex"
        expected = [2.5, 0, 4]
    offsets = cp.asarray([1, 3, 3, 4], dtype=cp.int64)
    result = cp.empty(3, dtype=dtype)
    kernel = CupyBackend.instance()[name, np.float64, np.float64, np.int64, np.float64]

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected device-to-host conversion")

    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", forbidden)
        kernel(result, values, offsets, 3, means)
    np.testing.assert_allclose(cp.asnumpy(result), expected)


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.uint64])
@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_integer_and_boolean_fused_powers(dtype, n):
    from awkward._backends.cupy import CupyBackend

    values = np.asarray([0, 1, 1, 0], dtype=dtype)
    offsets = cp.asarray([0, 2, 4], dtype=cp.int64)
    result = cp.empty(2, dtype=cp.float64)
    kernel = CupyBackend.instance()[
        "awkward_reduce_sumofpowers", np.float64, dtype, np.int64
    ]
    kernel(result, cp.asarray(values), offsets, 2, n)
    expected = (values.astype(np.float64) ** n).reshape(2, 2).sum(axis=1)
    np.testing.assert_array_equal(cp.asnumpy(result), expected)
