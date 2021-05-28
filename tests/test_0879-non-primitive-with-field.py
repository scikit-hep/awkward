# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    x = ak.Array({"x": np.arange(10)})

    xy = ak.with_field(base=x, what=None, where="y")

    # Try to access the type of a single element
    # This raises a ValueError in #879
    xy_type = xy.type  # noqa: F841
