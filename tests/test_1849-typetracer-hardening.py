# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test_list_array_pad_none_option():
    layout = ak.contents.ListArray(
        ak.index.Index64(np.array([0, 3, 6, 9])),
        ak.index.Index64(np.array([3, 6, 9, 11])),
        ak.contents.NumpyArray(np.arange(11)),
    )
    result = layout.pad_none(1, axis=-1, clip=False)
    assert ak.type(result) == ak.types.ListType(
        ak.types.OptionType(ak.types.NumpyType("int64"))
    )

    result_tt = layout.typetracer.pad_none(1, axis=-1, clip=False)
    assert ak.type(result_tt) == ak.types.ListType(
        ak.types.OptionType(ak.types.NumpyType("int64"))
    )
