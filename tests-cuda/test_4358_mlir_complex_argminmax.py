# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("cupy")
pytest.importorskip("numba_cuda_mlir")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("operation", [ak.argmin, ak.argmax], ids=["argmin", "argmax"])
def test_mlir_compiles_complex_device_helpers(dtype, operation):
    # Empty segments, equal real parts, equal values and a segment larger than
    # a CUDA block exercise calls from CCCL's MLIR operator into both helpers.
    small = np.array(
        [1 + 2j, 1 + 3j, 1 + 3j, -2 + 5j, -2 + 5j, 3 - 1j, 3 - 1j, 7], dtype=dtype
    )
    large = np.full(513, 2 + 3j, dtype=dtype)
    large[256] = 2 + 4j
    large[512] = -1 + 100j
    cpu = ak.unflatten(np.concatenate([small, large]), [0, 3, 4, 1, 513])
    gpu = ak.to_backend(cpu, "cuda")
    expected = operation(cpu, axis=-1, mask_identity=False)
    for _ in range(2):
        actual = operation(gpu, axis=-1, mask_identity=False)
        assert ak.backend(actual) == "cuda"
        assert ak.to_list(actual) == ak.to_list(expected)
    ak._connect.cuda.synchronize_cuda()
