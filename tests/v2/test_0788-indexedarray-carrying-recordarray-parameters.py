# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    one = ak._v2.highlevel.Array([[{"x": 1}], [], [{"x": 2}]], with_name="One")
    two = ak._v2.highlevel.Array([[{"x": 1.1}], [], [{"x": 2.2}]], with_name="Two")
    assert (
        str(
            ak._v2.operations.with_name(
                ak._v2.operations.concatenate([one, two], axis=1), "All"
            ).type
        )
        == "3 * var * All[x: float64]"
    )
    assert (
        str(
            ak._v2.operations.with_name(
                ak._v2.operations.concatenate([one[1:], two[1:]], axis=1),
                "All",
            ).type
        )
        == "2 * var * All[x: float64]"
    )
