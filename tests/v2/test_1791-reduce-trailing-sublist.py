# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    array = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(
                np.array(
                    [0, 1, 1],
                    dtype=np.int64,
                )
            ),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(
                    np.array(
                        [0, 1],
                        dtype=np.int64,
                    )
                ),
                ak._v2.contents.NumpyArray(np.arange(1)),
            ),
        )
    )

    reduced = ak._v2.sum(array, axis=1)
    # We currently get a ListArray here. Ensure that the start/stops are correct
    assert np.asarray(reduced.layout.starts).tolist() == [0, 1]
    assert np.asarray(reduced.layout.stops).tolist() == [1, 1]
