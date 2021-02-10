# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pyarrow = pytest.importorskip("pyarrow")


def test_categorical_is_valid():
    # validate a categorical array by its content
    arr = ak.Array([2019, 2020, 2021, 2020, 2019])
    categorical = ak.to_categorical(arr)
    assert ak.is_valid(categorical)


def test_optional_categorical_from_arrow():
    # construct categorical array from option-typed DictionaryArray
    indices = pyarrow.array([0, 1, 0, 1, 2, 0, 2])
    nan_indices = pyarrow.array([0, 1, 0, 1, 2, None, 0, 2])
    dictionary = pyarrow.array([2019, 2020, 2021])

    dict_array = pyarrow.DictionaryArray.from_arrays(indices, dictionary)
    categorical_array = ak.from_arrow(dict_array)
    assert categorical_array.layout.parameter("__array__") == "categorical"

    option_dict_array = pyarrow.DictionaryArray.from_arrays(nan_indices, dictionary)
    option_categorical_array = ak.from_arrow(option_dict_array)
    assert option_categorical_array.layout.parameter("__array__") == "categorical"
