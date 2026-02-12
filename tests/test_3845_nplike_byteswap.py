# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import warnings

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.typetracer import TypeTracer


def _require_backend(backend: str) -> None:
    if backend == "cuda":
        pytest.importorskip("cupy")
    elif backend == "jax":
        pytest.importorskip("jax")
        ak.jax.register_and_check()


def _get_nplike(backend: str):
    if backend == "cpu":
        return Numpy.instance()
    if backend == "cuda":
        from awkward._nplikes.cupy import Cupy

        return Cupy.instance()
    if backend == "jax":
        from awkward._nplikes.jax import Jax

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Jax.instance()
    raise AssertionError(f"unknown backend: {backend}")


def _to_numpy(backend: str, array):
    if backend == "cuda":
        cupy = pytest.importorskip("cupy")
        return cupy.asnumpy(array)
    else:
        return np.asarray(array)


DTYPE_CASES = [
    np.array([1, 2, 3], dtype=dtype)
    for dtype in [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        "datetime64[ns]",
        "timedelta64[ns]",
    ]
]

# Ensured each nplike backend matches NumPy byteswap byte-level semantics.


@pytest.mark.parametrize("backend", ["cpu", "cuda", "jax"])
@pytest.mark.parametrize("values", DTYPE_CASES)
def test_nplike_byteswap_matches_numpy(values: np.ndarray, backend: str):
    _require_backend(backend)
    nplike = _get_nplike(backend)

    try:
        array = nplike.asarray(values)
    except (TypeError, ValueError, NotImplementedError):
        pytest.skip(f"{backend} does not support dtype {values.dtype}")

    swapped = nplike.byteswap(array)
    got = _to_numpy(backend, swapped)
    expected = values.byteswap(inplace=False)

    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert got.tobytes() == expected.tobytes()


def test_typetracer_byteswap_errors():
    nplike = TypeTracer.instance()
    array = nplike.asarray(np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(
        NotImplementedError, match="TypeTracer does not support byteswap"
    ):
        nplike.byteswap(array)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "jax"])
def test_to_from_buffers_byteorder_roundtrip_all_backends(backend: str):
    _require_backend(backend)

    source = ak.to_backend(ak.Array([[1.0, 2.0], [], [3.0]]), backend)
    form, length, container = ak.to_buffers(source, byteorder=">")

    expected_offsets = ak._util.native_to_byteorder(
        np.array([0, 2, 2, 3], dtype=np.int64), ">"
    )
    expected_data = ak._util.native_to_byteorder(np.array([1.0, 2.0, 3.0]), ">")
    got_offsets = _to_numpy(backend, container["node0-offsets"])
    got_data = _to_numpy(backend, container["node1-data"])

    assert got_offsets.tobytes() == expected_offsets.tobytes()
    assert got_data.tobytes() == expected_data.tobytes()

    reconstructed = ak.from_buffers(
        form, length, container, backend=backend, byteorder=">"
    )
    assert ak.to_list(reconstructed) == ak.to_list(source)
    assert ak.backend(reconstructed) == backend


@pytest.mark.parametrize("backend", ["cpu", "cuda", "jax"])
def test_from_buffers_accepts_big_endian_bytes_all_backends(backend: str):
    _require_backend(backend)

    form = ak.forms.NumpyForm("float64", form_key="node0")
    container = {"node0-data": np.array([1.0, 2.0, 3.0], dtype=">f8").tobytes()}

    out = ak.from_buffers(form, 3, container, backend=backend, byteorder=">")
    assert ak.to_list(out) == [1.0, 2.0, 3.0]
    assert ak.backend(out) == backend
