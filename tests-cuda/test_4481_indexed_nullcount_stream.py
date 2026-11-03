# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("length", [0, 8, 65, 4097])
def test_nullcount_immediate_read_on_nonblocking_stream(dtype, length):
    from awkward._backends.cupy import CupyBackend

    source = np.arange(length, dtype=dtype)
    if dtype is not np.uint32:
        source[1::3] = -1
    expected = int(np.count_nonzero(source < 0))
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        index = cp.asarray(source)
        count = cp.empty(1, dtype=cp.int64)
        kernel = CupyBackend.instance()["awkward_IndexedArray_numnull", np.int64, dtype]
        for _ in range(20):
            count.fill(4294967297)
            kernel(count, index, length)
            # Same immediate scalar-read boundary as IndexedOptionArray.project.
            assert count[0].item() == expected
        ak._connect.cuda.synchronize_cuda(stream)


def test_missing_jagged_slice_projection_repeated():
    source = ak.Array([[0, 1, 2], [], [2, 3, 4], [7, 8]])
    selector = [[0, None, -1], [], [None, 1], [0, 1, None]]
    expected = source[selector].to_list()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        array = ak.to_backend(source, "cuda")
        for _ in range(20):
            result = array[selector]
            assert result.to_list() == expected
            assert ak.to_list(result.layout.content.project()) == [0, 2, 3, 7, 8]
        ak._connect.cuda.synchronize_cuda(stream)
