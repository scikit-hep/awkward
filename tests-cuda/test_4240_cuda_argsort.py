# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


INTEGER_DTYPES = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
]
FLOAT_DTYPES = ["float32", "float64"]


def _cpu_array(dtype):
    """A ragged array with empty lists, duplicates, and (for floats) NaNs."""
    if dtype == "bool":
        segs = [[True, False, True, True], [], [False, False], [True]]
        return ak.Array(segs)
    if dtype in FLOAT_DTYPES:
        nan = float("nan")
        # Finite values + NaN. (A real +-inf alongside NaN hits a documented
        # sentinel-collision edge case and is intentionally excluded here.)
        segs = [
            [3.0, nan, 1.0, 1.0],
            [],
            [nan, nan, 4.0],
            [2.0],
            [9.0, 0.0, 0.0, 7.0, 7.0],
        ]
        return ak.values_astype(ak.Array(segs), dtype)
    # signed/unsigned integers: keep values non-negative so uint is valid
    segs = [[3, 1, 2, 1], [], [5, 5, 4], [2], [9, 0, 0, 7, 7]]
    return ak.values_astype(ak.Array(segs), dtype)


def _nan_aware_equal(a, b):
    """Compare nested lists treating NaN == NaN."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _nan_aware_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


@pytest.mark.parametrize("dtype", ["bool", *INTEGER_DTYPES, *FLOAT_DTYPES])
@pytest.mark.parametrize("ascending", [True, False])
def test_argsort_matches_cpu_stable(dtype, ascending):
    # stable=True gives a deterministic permutation, so CUDA and CPU must return
    # the exact same indices (including NaN placement at the front).
    cpu = _cpu_array(dtype)
    gpu = ak.to_backend(cpu, "cuda")

    out_cpu = ak.argsort(cpu, axis=-1, ascending=ascending, stable=True)
    out_gpu = ak.argsort(gpu, axis=-1, ascending=ascending, stable=True)

    assert ak.to_list(ak.to_backend(out_gpu, "cpu")) == ak.to_list(out_cpu)


@pytest.mark.parametrize("dtype", ["bool", *INTEGER_DTYPES, *FLOAT_DTYPES])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
def test_argsort_carry_reproduces_sort(dtype, ascending, stable):
    # Regardless of tie-breaking, gathering with the argsort result must yield
    # the same ordering as ak.sort. Compare against the CPU backend (the source
    # of truth) so this validates the GPU permutation even when stable=False,
    # where exact indices may legitimately differ.
    cpu = _cpu_array(dtype)
    gpu = ak.to_backend(cpu, "cuda")

    carry = ak.argsort(gpu, axis=-1, ascending=ascending, stable=stable)
    via_carry = ak.to_list(ak.to_backend(gpu[carry], "cpu"))
    sorted_cpu = ak.to_list(ak.sort(cpu, axis=-1, ascending=ascending, stable=stable))

    assert _nan_aware_equal(via_carry, sorted_cpu)


@pytest.mark.parametrize("dtype", ["bool", *INTEGER_DTYPES, *FLOAT_DTYPES])
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_matches_cpu(dtype, ascending):
    # Cross-check the sibling awkward_sort (segmented_sort) path too.
    cpu = _cpu_array(dtype)
    gpu = ak.to_backend(cpu, "cuda")

    out_cpu = ak.to_list(ak.sort(cpu, axis=-1, ascending=ascending, stable=True))
    out_gpu = ak.to_list(
        ak.to_backend(ak.sort(gpu, axis=-1, ascending=ascending, stable=True), "cpu")
    )

    assert _nan_aware_equal(out_gpu, out_cpu)


def test_argsort_all_empty():
    gpu = ak.to_backend(ak.Array([[], [], []]), "cuda")
    out = ak.to_list(ak.to_backend(ak.argsort(gpu, axis=-1), "cpu"))
    assert out == [[], [], []]
