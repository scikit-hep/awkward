# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

REDUCERS = [ak.sum, ak.prod, ak.min, ak.max, ak.argmax, ak.count, ak.any]


def var():
    return ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3, 6, 7, 7, 9, 10, 12], dtype=np.int64)),
        ak.contents.NumpyArray(np.arange(12, dtype=np.float64)),
    )


LAYOUTS = {
    "list": ak.contents.RegularArray(var(), size=2),
    "option": ak.contents.RegularArray(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, -1, 2, 3, 4, -1, 6, 7], dtype=np.int64)),
            var(),
        ),
        size=2,
    ),
    "nested": ak.contents.RegularArray(ak.contents.RegularArray(var(), size=2), size=2),
}


@pytest.mark.parametrize(
    "forget_length", [False, True], ids=["known-length", "unknown-length"]
)
@pytest.mark.parametrize(
    ("name", "axis"),
    [("list", -1), ("option", -1), ("nested", -1), ("nested", -2)],
    ids=["list", "option", "nested", "nested-axis-2"],
)
def test_reduce_below_regular(name, axis, forget_length):
    layout = LAYOUTS[name]
    array = ak.Array(layout)
    typetracer = ak.Array(layout.to_typetracer(forget_length=forget_length))
    for reducer in REDUCERS:
        for keepdims in (False, True):
            expected = reducer(array, axis=axis, keepdims=keepdims)
            result = reducer(typetracer, axis=axis, keepdims=keepdims)
            assert result.layout.form == expected.layout.form
