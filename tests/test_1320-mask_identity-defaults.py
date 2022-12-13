# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([[1, 2], [], [-np.inf]])

    # monoidal reducers (have identities)
    assert ak.count(array, axis=1)[1] == 0
    assert ak.count_nonzero(array, axis=1)[1] == 0
    assert ak.sum(array, axis=1)[1] == 0
    assert ak.prod(array, axis=1)[1] == 1
    assert ak.all(array, axis=1)[1]
    assert not ak.any(array, axis=1)[1]

    # semigroup reducers (no identity; arguable in the case of min/max on floats)
    assert ak.max(array, axis=1)[1] is None
    assert ak.min(array, axis=1)[1] is None
    assert ak.argmax(array, axis=1)[1] is None
    assert ak.argmin(array, axis=1)[1] is None

    # defined in terms of reducers, with ak.count in the denominator
    assert np.isnan(ak.moment(array, 1, axis=1)[1])
    assert np.isnan(ak.mean(array, axis=1)[1])
    assert np.isnan(ak.var(array, axis=1)[1])
    assert np.isnan(ak.std(array, axis=1)[1])
    assert np.isnan(ak.corr(array, array, axis=1)[1])
    assert np.isnan(ak.covar(array, array, axis=1)[1])

    assert np.isnan(ak.linear_fit(array, array, axis=1)["intercept", 1])
    assert np.isnan(ak.linear_fit(array, array, axis=1)["slope", 1])
    assert np.isnan(ak.linear_fit(array, array, axis=1)["intercept_error", 1])
    assert np.isnan(ak.linear_fit(array, array, axis=1)["slope_error", 1])

    # defined in terms of reducers, but a map-like operation
    assert ak.softmax(array, axis=1)[1].to_list() == []
    assert np.isnan(ak.softmax(array, axis=1)[2, 0])

    # defined in terms of reducers, computed from ak.min and ak.max
    assert ak.ptp(array, axis=1)[1] is None
