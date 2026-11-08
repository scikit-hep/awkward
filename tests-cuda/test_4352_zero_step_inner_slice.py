# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import pytest

import awkward as ak


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp.cuda.Device().synchronize()
    cp._default_memory_pool.free_all_blocks()


# where the CPU kernels looped forever, the CUDA ones guard `step != 0` and
# quietly returned empty lists
@pytest.mark.parametrize(
    "where", [(slice(None, None, 0),), (slice(None), slice(None, None, 0))]
)
def test_zero_step_raises(where):
    array = ak.Array([[1, 2, 3], [4, 5, 6]], backend="cuda")
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        array[where]
