# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np  # noqa: F401

to_list = ak._v2.operations.to_list


def test_ufuncs_on_records_1439():
    array = ak._v2.Array([{"x": 1.0, "y": 1.1}, {"x": 2.0, "y": 2.2}])
    with pytest.raises(ValueError):
        np.absolute(array)


def test_this_should_raise_a_warning():
    one = ak._v2.Array([{"x": 1}, {"x": 2}, {"x": 3}])
    two = ak._v2.Array([{"x": 1.1}, {"x": 2.2}, {"x": 3.3}])
    with pytest.raises(ValueError):
        one + two


def test_this_should_not():
    def overload_add(left, right):
        return ak._v2.Array({"x": left.x + right.x})

    behavior = {}
    behavior[np.add, "Overload", "Overload"] = overload_add

    one = ak._v2.Array(
        [{"x": 1}, {"x": 2}, {"x": 3}], with_name="Overload", behavior=behavior
    )
    two = ak._v2.Array(
        [{"x": 1.1}, {"x": 2.2}, {"x": 3.3}], with_name="Overload", behavior=behavior
    )

    assert (one + two).tolist() == [{"x": 2.1}, {"x": 4.2}, {"x": 6.3}]
