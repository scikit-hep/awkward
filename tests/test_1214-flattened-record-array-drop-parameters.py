# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401
from awkward._v2.tmp_for_testing import v1_to_v2


def test():
    array = ak.Array(
        [[{"x": [[1, 2], []]}, {"x": []}, {"x": [[3]]}]], with_name="Record"
    )
    array_v2 = ak._v2.highlevel.Array(v1_to_v2(array.layout))

    # Flattening inside record does not drop parameters
    assert "__record__" not in ak.parameters(ak.flatten(array, axis=3))
    assert "__record__" not in array_v2.layout.flatten(axis=3).parameters

    # Flattening before record does not drop parameters
    assert "__record__" in ak.parameters(ak.flatten(array, axis=1))
    assert "__record__" in array_v2.layout.flatten(axis=1).parameters
