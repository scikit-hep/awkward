# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_raggedtensor",)


@high_level_function()
def from_raggedtensor(array):
    """
    Args:
        array: (`tensorflow.RaggedTensor`):
            RaggedTensor to convert into an  Awkward Array.

    Converts a TensorFlow RaggedTensor into an Awkward Array.

    If `array` contains any other data types the function raises an error.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        # get the flat values
        content = array.flat_values.numpy()
    except AttributeError as err:
        raise TypeError(
            """only RaggedTensor can be converted to awkward array"""
        ) from err
    # convert them to ak.contents right away
    content = ak.contents.NumpyArray(content)

    # get the offsets
    offsets_arr = []
    for splits in array.nested_row_splits:
        split = splits.numpy()
        # convert to ak.index
        offset = ak.index.Index64(split)
        offsets_arr.append(offset)

    # if a tensor has one *ragged dimension*
    if len(offsets_arr) == 1:
        result = ak.contents.ListOffsetArray(offsets_arr[0], content)
        return ak.Array(result)

    # if a tensor has multiple *ragged dimensions*
    return ak.Array(_recursive_call(content, offsets_arr, 0))


def _recursive_call(content, offsets_arr, count):
    if count == len(offsets_arr) - 2:
        return ak.contents.ListOffsetArray(
            offsets_arr[count],
            ak.contents.ListOffsetArray(offsets_arr[count + 1], content),
        )
    else:
        return ak.contents.ListOffsetArray(
            offsets_arr[count], _recursive_call(content, offsets_arr, count)
        )
