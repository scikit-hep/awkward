# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_fused_complex_kernel_cuda(dtype):
    data = np.array(
        [999j, 1 + 2j, complex(1, np.nan), complex(np.nan, 2), 3 + 4j, 999j],
        dtype=dtype,
    )
    component = np.float32 if dtype is np.complex64 else np.float64
    with cp.cuda.Stream(non_blocking=True):
        source = cp.asarray(data)
        sums = cp.full(5, 99 + 99j, dtype=dtype)
        counts = cp.full(5, -99, dtype=cp.int64)
        offsets = cp.asarray([1, 3, 3, 5], dtype=cp.int64)
        backend = ak.to_layout(source).backend
        backend.maybe_kernel_error(
            backend[
                "awkward_reduce_nansum_count_complex",
                component,
                np.int64,
                component,
                np.int64,
            ](sums.view(component), counts, source.view(component), offsets, 3)
        )
        ak._connect.cuda.synchronize_cuda()
        np.testing.assert_array_equal(
            cp.asnumpy(sums), [1 + 2j, 0j, 3 + 4j, 99 + 99j, 99 + 99j]
        )
        np.testing.assert_array_equal(cp.asnumpy(counts), [1, 0, 1, -99, -99])
