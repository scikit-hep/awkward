# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import warnings

import cupy as cp
import numpy as np
import pytest
from cupyx import allow_synchronize

import awkward as ak
from awkward._connect.cuda import _compute
from awkward._nplikes.cupy import Cupy


@pytest.mark.parametrize("values", [[1000.0, 1001.0], [[1000.0, 1001.0], [1002.0]]])
def test_softmax_maximum_computation_stays_on_device(values, monkeypatch):
    array = ak.to_backend(ak.Array(values), "cuda")
    expected = ak.to_list(ak.flatten(ak.softmax(ak.Array(values)), axis=None))
    maximum_impl = ak.operations.ak_max._impl
    nplike_maximum = Cupy.max
    segmented_reduce = _compute._segmented_reduce_raw

    # Populate compilation caches before checking the actual reduction.
    ak.softmax(array)
    cp.cuda.get_current_stream().synchronize()
    calls = []

    def checked_nplike_maximum(self, *args, **kwargs):
        result = nplike_maximum(self, *args, **kwargs)
        assert isinstance(result, cp.ndarray)
        return result

    monkeypatch.setattr(Cupy, "max", checked_nplike_maximum)

    def checked_segmented_maximum(**kwargs):
        assert kwargs.get("max_segment_size") is None
        return segmented_reduce(**kwargs)

    def guard_reducer(implementation):
        def checked_reducer(*args, **kwargs):
            # Scope this to numerical reduction: layout preparation has existing
            # scalar offset reads, independent of maximum-value/hint transfers.
            with warnings.catch_warnings(), monkeypatch.context() as patch:
                patch.setattr(
                    _compute, "_segmented_reduce_raw", checked_segmented_maximum
                )
                warnings.filterwarnings(
                    "ignore",
                    message="cupyx.allow_synchronize will be removed",
                    category=UserWarning,
                )
                with allow_synchronize(False):
                    return implementation(*args, **kwargs)

        return checked_reducer

    monkeypatch.setattr(
        ak._reducers.Max, "apply", guard_reducer(ak._reducers.Max.apply)
    )
    monkeypatch.setattr(
        ak._reducers.AxisNoneMax, "apply", guard_reducer(ak._reducers.AxisNoneMax.apply)
    )

    def checked_maximum(*args, **kwargs):
        result = maximum_impl(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(ak.operations.ak_max, "_impl", checked_maximum)
    result = ak.softmax(array)

    assert len(calls) == 1
    assert ak.backend(calls[0]) == "cuda"
    assert ak.backend(result) == "cuda"
    np.testing.assert_allclose(ak.to_list(ak.flatten(result, axis=None)), expected)
