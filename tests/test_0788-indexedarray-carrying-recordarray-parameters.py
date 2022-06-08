# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_feature():
    one = ak.Array([[{"x": 1}], [], [{"x": 2}]], with_name="One")
    two = ak.Array([[{"x": 1.1}], [], [{"x": 2.2}]], with_name="Two")
    assert (
        str(ak.with_name(ak.concatenate([one, two], axis=1), "All").type)
        == '3 * var * All["x": float64]'
    )
    assert (
        str(ak.with_name(ak.concatenate([one[1:], two[1:]], axis=1), "All").type)
        == '2 * var * All["x": float64]'
    )


def test_regression():
    def action():
        raise Exception("should not be called")

    form = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3])).form

    ak.layout.VirtualArray(ak.layout.ArrayGenerator(action, form=form, length=3))

    array = ak.Array(
        ak.layout.RecordArray(
            [
                ak.layout.VirtualArray(
                    ak.layout.ArrayGenerator(action, form=form, length=3)
                )
            ],
            ["x"],
        ),
    )

    ak.with_name(array, "Sam")
