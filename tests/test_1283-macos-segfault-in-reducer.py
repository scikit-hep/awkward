# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    v = ak.from_iter([[[1, 2, 3], [4, 5]], [[3, 4, 5], [6, 7]], [[5, 6, 7], [8, 9]]])
    M = np.asarray([[1, 1.4, -0.3], [1.4, 1, 1.2], [-0.3, 1.2, 1]])
    M_times_v = M[..., np.newaxis, np.newaxis] * v[np.newaxis, ...]
    v_times_M_times_v = v[:, np.newaxis, ...] * M_times_v

    # frequently (not always) segfaults with axis=0 or 1
    assert np.sum(v_times_M_times_v, axis=0).tolist() == [
        [[3.6999999999999993, 11.6, 23.699999999999996], []],
        [[39.99999999999999, 60.5, 0.0, 31.2, 56.0, 88.0], []],
        [
            [
                127.19999999999999,
                173.60000000000002,
                0.0,
                41.5,
                61.199999999999996,
                84.7,
            ],
            [],
        ],
    ]

    array = ak.from_iter(
        [[[[1], [4, 9]], [[5.6], [14, 25.2]]], [[[5.6], [14, 25.2]], [[16], [25, 36]]]]
    )

    # always(?) segfaults with axis=0 or 1, also when using np.sum
    assert ak.sum(array, axis=0).tolist() == [[[6.6], []], [[18.0, 34.2, 21.6], []]]
    assert ak.sum(array, axis=1).tolist() == [[[6.6], []], [[18.0, 34.2, 21.6], []]]
