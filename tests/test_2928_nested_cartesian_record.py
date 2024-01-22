# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pa = pytest.importorskip("pyarrow")


def test():
    one = ak.zip({"pt": [[1, 2], [1]], "eta": [[1.1, 2.1], [1]]})
    two = ak.zip({"pt": [[2.1], [2.21, 2.22]], "eta": [[2.11], [2.221, 2.222]]})
    assert ak.cartesian({"foo": one, "bar": two}, nested=False).layout.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 2, 4]),
            ak.contents.RecordArray(
                [
                    ak.contents.IndexedArray(
                        ak.index.Index64([0, 1, 2, 2]),
                        ak.contents.RecordArray(
                            [
                                ak.contents.NumpyArray(
                                    np.array([1, 2, 1], dtype=np.int64)
                                ),
                                ak.contents.NumpyArray(
                                    np.array([1.1, 2.1, 1.0], dtype=np.float64)
                                ),
                            ],
                            ["pt", "eta"],
                        ),
                    ),
                    ak.contents.IndexedArray(
                        ak.index.Index64([0, 0, 1, 2]),
                        ak.contents.RecordArray(
                            [
                                ak.contents.NumpyArray(
                                    np.array([2.1, 2.21, 2.22], dtype=np.float64)
                                ),
                                ak.contents.NumpyArray(
                                    np.array([2.11, 2.221, 2.222], dtype=np.float64)
                                ),
                            ],
                            ["pt", "eta"],
                        ),
                    ),
                ],
                ["foo", "bar"],
            ),
        )
    )
    assert ak.cartesian({"foo": one, "bar": two}, nested=True).layout.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 2, 3]),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 1, 2, 4]),
                ak.contents.RecordArray(
                    [
                        ak.contents.IndexedArray(
                            ak.index.Index64([0, 1, 2, 2]),
                            ak.contents.RecordArray(
                                [
                                    ak.contents.NumpyArray(
                                        np.array([1, 2, 1], dtype=np.int64)
                                    ),
                                    ak.contents.NumpyArray(
                                        np.array([1.1, 2.1, 1.0], dtype=np.float64)
                                    ),
                                ],
                                ["pt", "eta"],
                            ),
                        ),
                        ak.contents.IndexedArray(
                            ak.index.Index64([0, 0, 1, 2]),
                            ak.contents.RecordArray(
                                [
                                    ak.contents.NumpyArray(
                                        np.array([2.1, 2.21, 2.22], dtype=np.float64)
                                    ),
                                    ak.contents.NumpyArray(
                                        np.array([2.11, 2.221, 2.222], dtype=np.float64)
                                    ),
                                ],
                                ["pt", "eta"],
                            ),
                        ),
                    ],
                    ["foo", "bar"],
                ),
            ),
        )
    )
