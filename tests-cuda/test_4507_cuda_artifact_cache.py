# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32])
def test_persistent_artifact_rebinds_state_without_compilation(
    dtype, monkeypatch, tmp_path
):
    from awkward._connect.cuda import _compute_cache, _indexed_fill

    serialize, deserialize = _compute_cache._serialization_api()
    if serialize is None or deserialize is None:
        pytest.skip("Installed cuda.compute has no serialization API")
    monkeypatch.setenv("AWKWARD_CUDA_CACHE_DIR", str(tmp_path))
    _indexed_fill._setup.cache_clear()
    first = cp.asarray([0, 1, 2], dtype=dtype)
    output = cp.empty(3, dtype=cp.int64)
    _indexed_fill.fill(output, 0, first, 3, 7)
    np.testing.assert_array_equal(cp.asnumpy(output), [7, 8, 9])
    assert len(list(tmp_path.rglob("*.bin"))) == 1
    _indexed_fill._setup.cache_clear()

    def unexpected(*args, **kwargs):
        pytest.fail("A persisted fill rebuilt or recompiled its operator")

    monkeypatch.setattr(_indexed_fill, "make_unary_transform", unexpected)
    try:
        for base in (11, np.int64(22)):
            source = cp.asarray([0, 5, 1, 3], dtype=dtype)
            target = cp.full(9, 123, dtype=cp.int64)
            _indexed_fill.fill(target, 2, source, 4, base)
            cp.cuda.Stream.null.synchronize()
            expected = np.full(9, 123, dtype=np.int64)
            expected[2:6] = np.array([0, 5, 1, 3]) + base
            np.testing.assert_array_equal(cp.asnumpy(target), expected)
    finally:
        # Do not leave a deserialized object in another test's process cache.
        _indexed_fill._setup.cache_clear()
