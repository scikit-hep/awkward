# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    try:
        cp.cuda.Device().synchronize()  # wait for all kernels
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("GPU error during sync:", e)
    cp._default_memory_pool.free_all_blocks()


def test_nan_to_num_scalars():
    fmax = np.finfo(np.float64).max
    fmin = np.finfo(np.float64).min
    x = [np.inf, -np.inf, np.nan]
    expected = [fmax, fmin, 0.0]

    np_result = np.nan_to_num(np.array(x)).tolist()
    cp_result = cp.nan_to_num(cp.array(x)).tolist()
    cpu_result = to_list(ak.nan_to_num(ak.Array(x, backend="cpu")))
    cuda_result = to_list(ak.nan_to_num(ak.Array(x, backend="cuda")))

    assert np_result == expected
    assert cp_result == expected
    assert cpu_result == expected
    assert cuda_result == expected


def test_nan_to_num_default():
    fmax = np.finfo(np.float64).max
    fmin = np.finfo(np.float64).min
    x = [np.inf, -np.inf, np.nan, -128.0, 128.0]
    expected = [fmax, fmin, 0.0, -128.0, 128.0]

    np_result = np.nan_to_num(np.array(x)).tolist()
    cp_result = cp.nan_to_num(cp.array(x)).tolist()
    cpu_result = to_list(ak.nan_to_num(ak.Array(x, backend="cpu")))
    cuda_result = to_list(ak.nan_to_num(ak.Array(x, backend="cuda")))

    assert np_result == expected
    assert cp_result == expected
    assert cpu_result == expected
    assert cuda_result == expected


def test_nan_to_num_scalar_replacements():
    x = [np.inf, -np.inf, np.nan, -128.0, 128.0]
    expected = [33333333.0, 33333333.0, -9999.0, -128.0, 128.0]

    np_result = np.nan_to_num(
        np.array(x), nan=-9999, posinf=33333333, neginf=33333333
    ).tolist()
    cp_result = cp.nan_to_num(
        cp.array(x), nan=-9999, posinf=33333333, neginf=33333333
    ).tolist()
    cpu_result = to_list(
        ak.nan_to_num(
            ak.Array(x, backend="cpu"), nan=-9999, posinf=33333333, neginf=33333333
        )
    )
    cuda_result = to_list(
        ak.nan_to_num(
            ak.Array(x, backend="cuda"), nan=-9999, posinf=33333333, neginf=33333333
        )
    )

    assert np_result == expected
    assert cp_result == expected
    assert cpu_result == expected
    assert cuda_result == expected


def test_nan_to_num_array_replacements():
    x = [np.inf, -np.inf, np.nan, -128.0, 128.0]
    nan = [11, 12, -9999, 13, 14]
    posinf = [33333333, 11, 12, 13, 14]
    neginf = [11, 33333333, 12, 13, 14]
    expected = [33333333.0, 33333333.0, -9999.0, -128.0, 128.0]

    np_result = np.nan_to_num(
        np.array(x), nan=np.array(nan), posinf=np.array(posinf), neginf=np.array(neginf)
    ).tolist()
    cp_result = cp.nan_to_num(
        cp.array(x), nan=cp.array(nan), posinf=cp.array(posinf), neginf=cp.array(neginf)
    ).tolist()
    cpu_result = to_list(
        ak.nan_to_num(
            ak.Array(x, backend="cpu"),
            nan=ak.Array(nan, backend="cpu"),
            posinf=ak.Array(posinf, backend="cpu"),
            neginf=ak.Array(neginf, backend="cpu"),
        )
    )
    cuda_result = to_list(
        ak.nan_to_num(
            ak.Array(x, backend="cuda"),
            nan=ak.Array(nan, backend="cuda"),
            posinf=ak.Array(posinf, backend="cuda"),
            neginf=ak.Array(neginf, backend="cuda"),
        )
    )

    assert np_result == expected
    assert cp_result == expected
    assert cpu_result == expected
    assert cuda_result == expected


def test_nan_to_num_complex_default():
    fmax = np.finfo(np.float64).max
    y = [complex(np.inf, np.nan), complex(np.nan, 0), complex(np.nan, np.inf)]
    expected = [complex(fmax, 0.0), complex(0.0, 0.0), complex(0.0, fmax)]

    np_result = np.nan_to_num(np.array(y)).tolist()
    cp_result = cp.nan_to_num(cp.array(y)).tolist()
    cpu_result = to_list(ak.nan_to_num(ak.Array(y, backend="cpu")))
    cuda_result = to_list(ak.nan_to_num(ak.Array(y, backend="cuda")))

    assert np_result == expected
    assert cp_result == expected
    assert cpu_result == expected
    assert cuda_result == expected


def test_nan_to_num_complex_scalar_replacements():
    y = [complex(np.inf, np.nan), complex(np.nan, 0), complex(np.nan, np.inf)]
    expected = [complex(222222, 111111), complex(111111, 0), complex(111111, 222222)]

    np_result = np.nan_to_num(np.array(y), nan=111111, posinf=222222).tolist()
    cp_result = cp.nan_to_num(cp.array(y), nan=111111, posinf=222222).tolist()
    cpu_result = to_list(
        ak.nan_to_num(ak.Array(y, backend="cpu"), nan=111111, posinf=222222)
    )
    cuda_result = to_list(
        ak.nan_to_num(ak.Array(y, backend="cuda"), nan=111111, posinf=222222)
    )

    assert np_result == expected
    assert cp_result == expected
    assert cpu_result == expected
    assert cuda_result == expected


def test_nan_to_num_complex_array_replacements():
    y = [complex(np.inf, np.nan), complex(np.nan, 0), complex(np.nan, np.inf)]
    nan = [11, 12, 13]
    posinf = [21, 22, 23]
    neginf = [31, 32, 33]
    expected = [complex(21, 11), complex(12, 0), complex(13, 23)]

    np_result = np.nan_to_num(
        np.array(y), nan=np.array(nan), posinf=np.array(posinf), neginf=np.array(neginf)
    ).tolist()
    cp_result = cp.nan_to_num(
        cp.array(y), nan=cp.array(nan), posinf=cp.array(posinf), neginf=cp.array(neginf)
    ).tolist()
    cpu_result = to_list(
        ak.nan_to_num(
            ak.Array(y, backend="cpu"),
            nan=ak.Array(nan, backend="cpu"),
            posinf=ak.Array(posinf, backend="cpu"),
            neginf=ak.Array(neginf, backend="cpu"),
        )
    )
    cuda_result = to_list(
        ak.nan_to_num(
            ak.Array(y, backend="cuda"),
            nan=ak.Array(nan, backend="cuda"),
            posinf=ak.Array(posinf, backend="cuda"),
            neginf=ak.Array(neginf, backend="cuda"),
        )
    )

    assert np_result == expected
    assert cp_result == expected
    assert cpu_result == expected
    assert cuda_result == expected
