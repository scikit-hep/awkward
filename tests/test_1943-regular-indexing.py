# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_regular_index():
    array = ak.from_numpy(np.arange(4 * 4).reshape(4, 4))
    mask_regular = ak.Array((array > 4).layout.to_RegularArray())
    assert array[mask_regular].to_list() == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    mask_numpy = ak.to_numpy(mask_regular)
    assert array[mask_numpy].to_list() == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_non_list_index():
    array = ak.Array(
        [
            {"x": 10, "y": 1.0},
            {"x": 30, "y": 20.0},
            {"x": 40, "y": 20.0},
            {"x": "hi", "y": 20.0},
        ]
    )

    assert array[["x"]].to_list() == [{"x": 10}, {"x": 30}, {"x": 40}, {"x": "hi"}]

    fields_ak = ak.Array(["x"])
    assert array[fields_ak].to_list() == [{"x": 10}, {"x": 30}, {"x": 40}, {"x": "hi"}]

    fields_np = np.array(["x"])
    assert array[fields_np].to_list() == [{"x": 10}, {"x": 30}, {"x": 40}, {"x": "hi"}]

    class SizedIterable:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter(["y"])

    fields_custom = SizedIterable()
    assert array[fields_custom].to_list() == [
        {"y": 1.0},
        {"y": 20.0},
        {"y": 20.0},
        {"y": 20.0},
    ]

    fields_tuple = ("x",)
    assert array[fields_tuple].to_list() == [10, 30, 40, "hi"]
