# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward._connect.cuda as cuda
from awkward._kernels import CudaComputeKernel

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("fail", [False, True])
def test_nondefault_stream_status(fail):
    stream = cp.cuda.Stream(non_blocking=True)

    def decode(value):
        row, attempt = value
        return (
            None if row == -1 else ValueError(f"first row {row}, attempted {attempt}")
        )

    def impl():
        status = cp.empty(2, dtype=cp.int64)
        status[0] = 7 if fail else -1
        status[1] = 123
        cuda.defer_status(status, decode, stream=stream)

    with stream:
        CudaComputeKernel(impl, ("deferred_test",))()
        assert len(cuda.cuda_streamptr_to_contexts[stream.ptr][1]) == 1
        if fail:
            with pytest.raises(ValueError, match="first row 7, attempted 123"):
                cuda.synchronize_cuda(stream)
        else:
            cuda.synchronize_cuda(stream)
        assert cuda.cuda_streamptr_to_contexts[stream.ptr][1] == []
