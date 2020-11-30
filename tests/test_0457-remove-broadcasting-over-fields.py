# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_this_should_raise_a_warning():
    ak.deprecations_as_errors = True

    one = ak.Array([{"x": 1}, {"x": 2}, {"x": 3}])
    two = ak.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}])

    with pytest.raises(ValueError):
        one + two


def test_this_should_not():
    def overload_add(left, right):
        return ak.Array({"x": left.x + right.x})

    behavior = {}
    behavior[np.add, "Overload", "Overload"] = overload_add

    one = ak.Array(
        [{"x": 1}, {"x": 2}, {"x": 3}], with_name="Overload", behavior=behavior
    )
    two = ak.Array(
        [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}], with_name="Overload", behavior=behavior
    )

    assert (one + two).tolist() == [{"x": 2.1}, {"x": 4.2}, {"x": 6.3}]
