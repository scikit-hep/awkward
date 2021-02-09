# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    # None in the argument list is the case that breaks Awkward 1.1.0.
    ak.virtual(lambda arg: ak.Array([1, 2, 3]), args=(None,))
