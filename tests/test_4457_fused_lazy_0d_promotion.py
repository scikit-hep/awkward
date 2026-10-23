# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
import awkward._connect.cpu._fusion_codegen as cpu_fc
import awkward._connect.cuda._fusion_codegen as cuda_fc
from awkward._connect import cpu
from awkward._connect.lazy._fusion import py_scalar_literal


def test_0d_array_operands_match_eager_promotion():
    # Issue #4457: 0-d array operands must promote dtypes identically in fused,
    # unfused, and eager evaluation under NEP 50 (NumPy 2).
    a_u8 = ak.values_astype(ak.Array([[1, 2], [3]]), np.uint8)
    a_f32 = ak.values_astype(ak.Array([[1.0, 2.0], [3.0]]), np.float32)

    cases = [
        (a_u8, np.array(5, dtype=np.int64), np.int64, [[6, 7], [8]]),
        (a_f32, np.array(1.5, dtype=np.float64), np.float64, [[2.5, 3.5], [4.5]]),
        (a_f32, np.array(1.5, dtype=np.float32), np.float32, [[2.5, 3.5], [4.5]]),
        (a_u8, np.array(5, dtype=np.uint16), np.uint16, [[6, 7], [8]]),
    ]

    for arr, s, expected_dtype, expected_values in cases:
        eager = arr + s
        fused = (cpu.lazy(arr) + s).compute(fuse=True)
        unfused = (cpu.lazy(arr) + s).compute(fuse=False)

        assert eager.layout.content.dtype == expected_dtype
        assert fused.layout.content.dtype == expected_dtype
        assert unfused.layout.content.dtype == expected_dtype

        assert ak.to_list(eager) == expected_values
        assert ak.to_list(fused) == expected_values
        assert ak.to_list(unfused) == expected_values


def test_numpy_scalar_operands_match_eager_promotion():
    # NumPy scalars passed directly into the lazy pipeline
    a_u8 = ak.values_astype(ak.Array([[1, 2], [3]]), np.uint8)
    a_f32 = ak.values_astype(ak.Array([[1.0, 2.0], [3.0]]), np.float32)

    cases = [
        (a_u8, np.int64(10), np.int64, [[11, 12], [13]]),
        (a_f32, np.float64(2.5), np.float64, [[3.5, 4.5], [5.5]]),
        (a_f32, np.float32(2.5), np.float32, [[3.5, 4.5], [5.5]]),
        (a_u8, np.uint16(10), np.uint16, [[11, 12], [13]]),
        (a_u8, np.bool_(True), np.uint8, [[2, 3], [4]]),
    ]

    for arr, s, expected_dtype, expected_values in cases:
        eager = arr + s
        fused = (cpu.lazy(arr) + s).compute(fuse=True)
        unfused = (cpu.lazy(arr) + s).compute(fuse=False)

        assert eager.layout.content.dtype == expected_dtype
        assert fused.layout.content.dtype == expected_dtype
        assert unfused.layout.content.dtype == expected_dtype

        assert ak.to_list(eager) == expected_values
        assert ak.to_list(fused) == expected_values
        assert ak.to_list(unfused) == expected_values


def test_python_scalars_preserve_weak_promotion():
    # Python scalars must retain weak promotion under NEP 50 across all paths
    a_u8 = ak.values_astype(ak.Array([[1, 2], [3]]), np.uint8)
    a_f32 = ak.values_astype(ak.Array([[1.0, 2.0], [3.0]]), np.float32)

    cases = [
        (a_u8, 5, np.uint8, [[6, 7], [8]]),
        (a_f32, 1.5, np.float32, [[2.5, 3.5], [4.5]]),
        (a_u8, True, np.uint8, [[2, 3], [4]]),
    ]

    for arr, s, expected_dtype, expected_values in cases:
        eager = arr + s
        fused = (cpu.lazy(arr) + s).compute(fuse=True)
        unfused = (cpu.lazy(arr) + s).compute(fuse=False)

        assert eager.layout.content.dtype == expected_dtype
        assert fused.layout.content.dtype == expected_dtype
        assert unfused.layout.content.dtype == expected_dtype

        assert ak.to_list(eager) == expected_values
        assert ak.to_list(fused) == expected_values
        assert ak.to_list(unfused) == expected_values


def test_non_finite_numpy_scalars():
    f32 = ak.values_astype(ak.Array([[1.0, 2.0], [3.0]]), np.float32)

    # float32 with np.float64 non-finites promotes to float64
    fused_inf64 = (cpu.lazy(f32) + np.float64("inf")).compute(fuse=True)
    assert fused_inf64.layout.content.dtype == np.float64
    assert np.all(np.isinf(fused_inf64))

    # float32 with np.float32 non-finites stays float32
    fused_inf32 = (cpu.lazy(f32) + np.float32("inf")).compute(fuse=True)
    assert fused_inf32.layout.content.dtype == np.float32
    assert np.all(np.isinf(fused_inf32))

    # float32 with np.float32 nan stays float32
    fused_nan32 = (cpu.lazy(f32) + np.float32("nan")).compute(fuse=True)
    assert fused_nan32.layout.content.dtype == np.float32
    assert np.all(np.isnan(fused_nan32))


def test_py_scalar_literal_numpy_scalars():
    assert py_scalar_literal(np.int64(42)) == "np.int64(42)"
    assert py_scalar_literal(np.float32(1.5)) == "np.float32(1.5)"
    assert py_scalar_literal(np.float64(1.5)) == "np.float64(1.5)"
    assert py_scalar_literal(np.float32("inf")) == "np.float32('inf')"
    assert py_scalar_literal(np.float32("-inf")) == "np.float32('-inf')"
    assert py_scalar_literal(np.float32("nan")) == "np.float32('nan')"
    assert py_scalar_literal(np.float64("inf")) == "np.float64('inf')"
    assert py_scalar_literal(np.float64("-inf")) == "np.float64('-inf')"
    assert py_scalar_literal(np.float64("nan")) == "np.float64('nan')"


def test_classify_leaf_preserves_dtype_and_scalar_types():
    for val, expected_type, expected_dtype in [
        (np.array(5, dtype=np.int64), np.int64, np.dtype(np.int64)),
        (np.array(1.5, dtype=np.float32), np.float32, np.dtype(np.float32)),
        (np.int64(42), np.int64, np.dtype(np.int64)),
        (np.float32(3.14), np.float32, np.dtype(np.float32)),
    ]:
        k_cpu, p_cpu = cpu_fc._classify_leaf(val)
        assert k_cpu == "scalar"
        assert isinstance(p_cpu, expected_type)
        assert p_cpu.dtype == expected_dtype

        k_cuda, p_cuda = cuda_fc._classify_leaf(val)
        assert k_cuda == "scalar"
        assert isinstance(p_cuda, expected_type)
        assert p_cuda.dtype == expected_dtype


def test_cuda_infer_out_dtype_with_numpy_scalars():
    # Verify CUDA out_dtype inference without needing a physical GPU
    ns = {}
    exec(
        "def _fused_op(t):\n    return t + np.int64(5)\n",
        cuda_fc._EXEC_GLOBALS,
        ns,
    )
    op = ns["_fused_op"]
    col_u8 = [ak.values_astype(ak.Array([[1, 2]]), np.uint8)]
    dtype_inferred = cuda_fc._infer_out_dtype(op, col_u8, single=True)
    assert dtype_inferred == np.int64
