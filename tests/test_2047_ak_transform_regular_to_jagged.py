# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.from_numpy(np.arange(2 * 3 * 4 * 5, dtype=np.int64).reshape(2, 3, 4, 5))
    assert array.type == ak.types.ArrayType(
        ak.types.RegularType(
            ak.types.RegularType(
                ak.types.RegularType(ak.types.NumpyType("int64"), 5), 4
            ),
            3,
        ),
        2,
    )

    def apply(layout, **kwargs):
        return

    result_regular = ak.transform(
        apply,
        array,
        numpy_to_regular=True,
    )
    assert result_regular.type == ak.types.ArrayType(
        ak.types.RegularType(
            ak.types.RegularType(
                ak.types.RegularType(ak.types.NumpyType("int64"), 5), 4
            ),
            3,
        ),
        2,
    )

    result_ragged = ak.transform(
        apply,
        array,
        regular_to_jagged=True,
        numpy_to_regular=True,
    )
    assert result_ragged.type == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.ListType(ak.types.ListType(ak.types.NumpyType("int64")))
        ),
        2,
    )
