# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_toRegularArray():
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 5, 7, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 5 * 7 * 11 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 5, 7, 11, 0), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 5 * 7 * 11 * 0 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 5, 7, 0, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 5 * 7 * 0 * 11 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 5, 0, 7, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 5 * 0 * 7 * 11 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 0, 5, 7, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 0 * 5 * 7 * 11 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 0, 3, 5, 7, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 0 * 3 * 5 * 7 * 11 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((0, 2, 3, 5, 7, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "0 * 2 * 3 * 5 * 7 * 11 * float64"
    )

    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 5, 7, 0, 11, 0), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 5 * 7 * 0 * 11 * 0 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 5, 0, 7, 11, 0), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 5 * 0 * 7 * 11 * 0 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 3, 0, 5, 7, 0, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 3 * 0 * 5 * 7 * 0 * 11 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((2, 0, 3, 5, 7, 0, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "2 * 0 * 3 * 5 * 7 * 0 * 11 * float64"
    )
    assert (
        str(
            ak.type(
                ak.Array(
                    ak.from_numpy(
                        np.empty((0, 2, 3, 5, 7, 0, 11), np.float64)
                    ).layout.toRegularArray()
                )
            )
        )
        == "0 * 2 * 3 * 5 * 7 * 0 * 11 * float64"
    )


def test_actual():
    x = ak.from_numpy(np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4))
    s = x[..., :0]
    result = ak.zip({"q": s, "t": s})
    assert str(ak.type(result)) == '2 * 3 * 0 * {"q": int64, "t": int64}'
