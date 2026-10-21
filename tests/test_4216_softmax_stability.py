# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np

import awkward as ak


def test_softmax_is_stable_for_large_finite_values():
    array = ak.Array([[1000.0, 1001.0], [-1000.0, -1001.0]])

    result = ak.softmax(array, axis=-1)

    np.testing.assert_allclose(
        ak.to_numpy(result),
        [[0.26894142, 0.73105858], [0.73105858, 0.26894142]],
    )

    with_empty_list = ak.Array([[1000.0, 1001.0], []])
    assert ak.softmax(with_empty_list, axis=-1, mask_identity=True).to_list()[1] is None
