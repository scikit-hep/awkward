# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward.typetracer import (
    empty_if_typetracer,
    length_one_if_typetracer,
    typetracer_with_report,
)


@pytest.mark.parametrize("function", [empty_if_typetracer, length_one_if_typetracer])
def test_typetracer(function):
    def func(array):
        assert ak.backend(array) == "typetracer"

        radius = np.sqrt(array.x**2 + array.y**2)
        radius = function(radius)
        assert ak.backend(radius) == "cpu"
        if function is empty_if_typetracer:
            assert len(radius) == 0
        else:
            assert len(radius) == 1

        hist_contents, hist_edges = np.histogram(ak.flatten(radius, axis=None))

        return hist_contents

    array = ak.zip(
        {
            "x": [[0.2, 0.3, 0.4], [1, 2, 3], [1, 1, 2]],
            "y": [[0.1, 0.1, 0.2], [3, 1, 2], [2, 1, 2]],
            "z": [[0.1, 0.1, 0.2], [3, 1, 2], [2, 1, 2]],
        }
    )
    layout = ak.to_layout(array)
    form = layout.form_with_key("node{id}")

    meta, report = typetracer_with_report(form)
    meta = ak.Array(meta)

    func(meta)
    assert report.data_touched == ["node0", "node2", "node3"]


@pytest.mark.parametrize("regulararray", [False, True])
def test_multiplier(regulararray):
    a = ak.from_numpy(np.arange(2 * 3 * 5).reshape(2, 3, 5), regulararray=regulararray)
    assert str(a.type) == "2 * 3 * 5 * int64"

    b = a.layout.form.length_one_array()
    assert str(b.type) == "1 * 3 * 5 * int64"
    assert b.tolist() == [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
