# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert ak.to_list(
        ak.layout.NumpyArray(np.array([[1, 2], [3, 4]], np.int64)).numbers_to_type(
            "int16"
        )
    ) == [[1, 2], [3, 4]]
