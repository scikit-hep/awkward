# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_reducers():
    array = ak.Array([[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}]])

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.all(array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.any(array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.argmax(array, axis=1)) == [
            {"phi": True, "rho": True}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.argmin(array, axis=1)) == [{"phi": 0, "rho": 0}]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.count_nonzero(array, axis=1)) == [
            {"phi": 2, "rho": 2}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.count(array, axis=1)) == [{"phi": 2, "rho": 2}]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.max(array, axis=1)) == [
            {"phi": 0.1, "rho": 1.1}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.min(array, axis=1)) == [
            {"phi": -0.1, "rho": -1.1}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.prod(array, axis=1)) == [
            {"phi": -0.010000000000000002, "rho": -1.2100000000000002}
        ]

    with pytest.raises(TypeError):
        assert ak.to_list(ak.operations.sum(array, axis=1)) == [
            {"phi": 0.0, "rho": 0.0}
        ]


def test_overloaded_reducers():
    def overload_add(array, mask):
        return ak.contents.RecordArray(
            [
                ak.sum(array["rho"], axis=-1, mask_identity=False, highlevel=False),
                ak.sum(array["phi"], axis=-1, mask_identity=False, highlevel=False),
            ],
            ["rho", "phi"],
        )

    behavior = {}
    behavior[ak.sum, "VectorArray2D"] = overload_add

    array = ak.Array(
        [[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}]],
        with_name="VectorArray2D",
        behavior=behavior,
    )

    assert ak.to_list(ak.sum(array, axis=1)) == [{"rho": 0, "phi": 0}]

    with pytest.raises(
        TypeError, match=r"overloads for custom types: VectorArray2D, int"
    ):
        ak.to_list(array + 1)

    def overload_add2(array, mask):
        return ak.contents.RecordArray(
            [ak.contents.NumpyArray(np.asarray([2.4, 3, 4.5, 6], dtype=np.float64))],
            ["rho"],
        )

    behavior = {}
    behavior[ak.sum, "VectorArray2D"] = overload_add2

    array = ak.Array(
        [[{"rho": -1.1, "phi": -0.1}, {"rho": 1.1, "phi": 0.1}]],
        with_name="VectorArray2D",
        behavior=behavior,
    )

    assert ak.to_list(ak.sum(array, axis=1)) == [{"rho": 2.4}]

    array = ak.highlevel.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index8(np.array([True, True])),
            ak.contents.RecordArray(
                [
                    ak.contents.NumpyArray(np.arange(2)),
                    ak.contents.IndexedOptionArray(
                        ak.index.Index64(np.array([0, -1])),
                        ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
                    ),
                ],
                ["a", "b"],
            ),
            valid_when=True,
        )
    )

    with pytest.raises(TypeError, match=r"overloads for custom types: a, b"):
        ak.to_list(ak.sum(array, axis=0))

    def overload_argmax(array, mask):
        return ak.argmax(array["rho"], axis=-1, mask_identity=False, highlevel=False)

    behavior = {}
    behavior[ak.argmax, "VectorArray2D"] = overload_argmax

    array = ak.Array(
        [
            [
                {"rho": -1.1, "phi": -0.1},
                {"rho": 1.1, "phi": 0.1},
                {"rho": -1.1, "phi": 0.3},
            ],
            [
                {"rho": 1.1, "phi": 0.1},
                {"rho": -1.1, "phi": 0.3},
                {"rho": -1.1, "phi": -0.1},
            ],
        ],
        with_name="VectorArray2D",
        behavior=behavior,
    )

    assert ak.to_list(ak.argmax(array, axis=1)) == [1, 0]
