# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test():
    a = ak.Array([{"this": 100}])
    b = ak.Array([{"this": 90, "that": 100}])
    c = ak.operations.concatenate((a, b))

    with pytest.raises(ValueError):
        ak.operations.unzip(c)

    a = ak.Array([{"this": 100}])
    b = ak.Array([{"this": 90}])
    c = ak.operations.concatenate((a, b))

    (tmp,) = ak.operations.unzip(c)

    assert tmp.to_list() == [100, 90]


def test_zip_and_unzip():
    x = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])
    y = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    one = ak.operations.zip({"x": x, "y": y})
    two = ak.operations.zip({"x": x, "y": y}, depth_limit=1)
    xx, yy = ak.operations.unzip(two)
    assert isinstance(one.layout, ak.contents.Content)
    assert isinstance(two.layout, ak.contents.Content)
    assert isinstance(xx.layout, ak.contents.Content)
    assert isinstance(yy.layout, ak.contents.Content)
    assert to_list(one) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 1.1}, {"x": 3, "y": 1.1}],
        [],
        [{"x": 4, "y": 3.3}, {"x": 5, "y": 3.3}],
        [{"x": 6, "y": 4.4}],
        [
            {"x": 7, "y": 5.5},
            {"x": 8, "y": 5.5},
            {"x": 9, "y": 5.5},
            {"x": 10, "y": 5.5},
        ],
    ]
    assert to_list(two) == [
        {"x": [1, 2, 3], "y": 1.1},
        {"x": [], "y": 2.2},
        {"x": [4, 5], "y": 3.3},
        {"x": [6], "y": 4.4},
        {"x": [7, 8, 9, 10], "y": 5.5},
    ]
    assert to_list(xx) == [[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]]
    assert to_list(yy) == [1.1, 2.2, 3.3, 4.4, 5.5]
