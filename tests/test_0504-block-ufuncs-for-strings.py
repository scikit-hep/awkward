# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    ak.deprecations_as_errors = True

    array = ak.to_categorical(ak.Array([321, 1.1, 123, 1.1, 999, 1.1, 2.0]))
    assert ak.to_list(array * 10) == [3210, 11, 1230, 11, 9990, 11, 20]

    array = ak.Array(["HAL"])
    with pytest.raises(ValueError):
        array + 1
