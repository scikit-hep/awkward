# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward._v2 as ak  # noqa: F401


def test():
    a = ak.contents.RecordArray(
        [
            ak.contents.RegularArray(
                ak.contents.RegularArray(
                    ak.contents.NumpyArray(np.arange(4 * 3 * 2)), 4
                ),
                3,
            )
        ],
        None,
        2,
        parameters={"__record__": "recordname"},
    )
    assert a.parameters.get("__record__") == "recordname"
    b = a.flatten(axis=2)

    assert "__record__" not in b.parameters
