# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.arange(10)),
        ak.contents.RecordArray(
            [
                ak.contents.IndexedOptionArray(
                    ak.index.Index64(
                        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1], dtype=np.int64)
                    ),
                    ak.contents.NumpyArray(np.arange(10)),
                )
            ],
            ["x"],
        ),
    )
    assert ak.almost_equal(
        ak.fill_none(layout, 9, axis=0), ak.zip({"x": np.arange(10)})
    )
