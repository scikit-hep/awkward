# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_reducers():
    array = ak._v2.Array([[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}]])

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.all(array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.any(array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.argmax(array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.argmin(array, axis=1)) == [
            {"phi": 0, "rho": 0}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.count_nonzero(array, axis=1)) == [
            {"phi": 2, "rho": 2}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.count(array, axis=1)) == [
            {"phi": 2, "rho": 2}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.max(array, axis=1)) == [
            {"phi": 0.1, "rho": 1.1}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.min(array, axis=1)) == [
            {"phi": -0.1, "rho": -1.1}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.prod(array, axis=1)) == [
            {"phi": -0.010000000000000002, "rho": -1.2100000000000002}
        ]

    with pytest.raises(TypeError):
        assert ak._v2.to_list(ak._v2.operations.sum(array, axis=1)) == [
            {"phi": 0.0, "rho": 0.0}
        ]


def test_overloaded_reducers():
    def overload_add(array, axis=-1):
        return ak._v2.contents.RecordArray(
            [
                ak._v2.contents.NumpyArray(
                    np.asarray([ak._v2.sum(array["rho"], axis=axis)])
                ),
                ak._v2.contents.NumpyArray(
                    np.asarray([ak._v2.sum(array["phi"], axis=axis)])
                ),
            ],
            ["rho", "phi"],
        )

    behavior = {}
    behavior[ak._v2.sum, "VectorArray2D"] = overload_add

    array = ak._v2.Array(
        [[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}]],
        with_name="VectorArray2D",
        behavior=behavior,
    )

    with pytest.raises(NotImplementedError):
        assert ak._v2.to_list(ak._v2.sum(array, axis=1)) == [{"rho": 0, "phi": 0}]

    with pytest.raises(TypeError):
        ak._v2.to_list(array + 1)

    def overload_add2(array):
        return ak._v2.contents.RecordArray(
            [ak._v2.contents.NumpyArray(np.asarray([2.4, 3, 4.5, 6]))], ["rho"]
        )

    behavior = {}
    behavior[ak._v2.sum, "VectorArray2D"] = overload_add2

    array = ak._v2.Array(
        [[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}]],
        with_name="VectorArray2D",
        behavior=behavior,
    )

    with pytest.raises(NotImplementedError):
        assert ak._v2.to_list(ak._v2.sum(array, axis=1)) == [{"rho": 2.4}]

    array = ak._v2.highlevel.Array(
        ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index8(np.array([True, True])),
            ak._v2.contents.RecordArray(
                [
                    ak._v2.contents.NumpyArray(np.arange(2)),
                    ak._v2.contents.IndexedOptionArray(
                        ak._v2.index.Index64(np.array([0, -1])),
                        ak._v2.contents.NumpyArray(np.array([1], dtype=np.int64)),
                    ),
                ],
                ["a", "b"],
            ),
            valid_when=True,
        )
    )

    with pytest.raises(TypeError):
        ak._v2.to_list(ak._v2.sum(array, axis=0))
