# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import gc
import weakref

import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


def test_successful_slicing_does_not_format_or_retain_arrays(monkeypatch):
    import awkward._connect.cuda as cuda

    array = ak.to_backend(
        ak.to_regular([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], axis=-1), "cuda"
    )
    indices = ak.to_backend([[0], [1]], "cuda")
    refs = [weakref.ref(array), weakref.ref(indices)]

    def forbidden(*args, **kwargs):
        raise AssertionError("successful CUDA slicing must not format values")

    with monkeypatch.context() as patch:
        patch.setattr(ak.Array, "_repr", forbidden)
        result = array[:, indices]
    del array, indices
    gc.collect()
    assert all(ref() is None for ref in refs)
    cuda.synchronize_cuda(cp.cuda.get_current_stream())
    assert ak.to_list(result) == [[[1], [4]], [[5], [8]]]
