# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")

_compute = pytest.importorskip("awkward._connect.cuda._compute")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("awkward_reduce_sum", [3, 0, 12]),
        ("awkward_reduce_prod", [2, 1, 60]),
        ("awkward_reduce_min", [1, np.iinfo(np.int64).max, 3]),
        ("awkward_reduce_max", [2, np.iinfo(np.int64).min, 5]),
        ("awkward_reduce_countnonzero", [2, 0, 3]),
    ],
)
def test_repeated_reduction_without_device_derived_hint(monkeypatch, name, expected):
    data = cp.asarray([1, 2, 3, 4, 5], dtype=cp.int64)
    offsets = cp.asarray([0, 2, 2, 5], dtype=cp.int64)
    result = cp.empty(3, dtype=cp.int64)
    reduce = getattr(_compute, name)
    extra = (
        (expected[1],) if name in ("awkward_reduce_min", "awkward_reduce_max") else ()
    )
    # Compile and initialize CCCL before checking steady-state launches.
    reduce(result, data, offsets, 3, *extra)
    cp.cuda.get_current_stream().synchronize()
    raw_reduce = _compute._segmented_reduce_raw
    hints = []

    def checked_reduce(**kwargs):
        # Deriving this optional dispatch hint from offsets was the source of
        # the unwanted host synchronization. Check the actual CCCL boundary.
        assert kwargs["max_segment_size"] is None
        hints.append(kwargs["max_segment_size"])
        return raw_reduce(**kwargs)

    monkeypatch.setattr(_compute, "_segmented_reduce_raw", checked_reduce)
    for _ in range(5):
        reduce(result, data, offsets, 3, *extra)
    assert hints == [None] * 5
    np.testing.assert_array_equal(cp.asnumpy(result), expected)
