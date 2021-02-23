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


def test_categorical_from_arrow_ChunkedArray():
    indices = [0, 1, 0, 1, 2, 0, 2]
    indices_new_schema = [0, 1, 0, 1, 0]

    dictionary = pyarrow.array([2019, 2020, 2021])
    dictionary_new_schema = pyarrow.array([2019, 2020])

    dict_array = pyarrow.DictionaryArray.from_arrays(pyarrow.array(indices), dictionary)
    dict_array_new_schema = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array(indices_new_schema), dictionary_new_schema
    )

    batch = pyarrow.RecordBatch.from_arrays([dict_array], ["year"])
    batch_new_schema = pyarrow.RecordBatch.from_arrays(
        [dict_array_new_schema], ["year"]
    )

    batches = [batch] * 3
    batches_mixed_schema = [batch] + [batch_new_schema]

    table = pyarrow.Table.from_batches(batches)
    table_mixed_schema = pyarrow.Table.from_batches(batches_mixed_schema)

    array = ak.from_arrow(table)
    array_mixed_schema = ak.from_arrow(table_mixed_schema)

    assert np.asarray(array.layout.field(0).content.index).tolist() == indices * 3
    assert (
        np.asarray(array_mixed_schema.layout.field(0).content.index).tolist()
        == indices + indices_new_schema
    )
