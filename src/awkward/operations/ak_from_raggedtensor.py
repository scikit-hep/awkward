# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import re

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
        content = array.flat_values
    except AttributeError as err:
        raise TypeError(
            """only RaggedTensor can be converted to awkward array"""
        ) from err

    # handle gpu and cpu instances separately
    device = content.backing_device

    content = _tensor_to_np_or_cp(content, device)

    # convert flat_values to ak.contents right away
    content = ak.contents.NumpyArray(content)

    # get the offsets
    offsets_arr = []
    for splits in array.nested_row_splits:
        # handle gpu and cpu instances separately
        split = _tensor_to_np_or_cp(splits, device)
        # convert to ak.index
        offset = ak.index.Index64(split)
        offsets_arr.append(offset)

    # if a tensor has one *ragged dimension*
    if len(offsets_arr) == 1:
        result = ak.contents.ListOffsetArray(offsets_arr[0], content)
        return ak.Array(result)

    # if a tensor has multiple *ragged dimensions*
    return ak.Array(_recursive_call(content, offsets_arr, 0))


def _tensor_to_np_or_cp(array, device):
    matched_device = re.match(".*:(CPU|GPU):[0-9]+", device)

    if matched_device is None:
        raise NotImplementedError(
            f"TensorFlow device has an unexpected format: {device!r}"
        )
    elif matched_device.groups()[0] == "GPU":
        try:
            import tensorflow as tf
        except ImportError as err:
            raise ImportError(
                """to use ak.from_raggedtensor, you must install the 'tensorflow' package with:

            pip install tensorflow
    or
            conda install tensorflow"""
            ) from err

        from awkward._nplikes.cupy import Cupy

        cp = Cupy.instance()
        return cp.from_dlpack(tf.experimental.dlpack.to_dlpack(array))
    elif matched_device.groups()[0] == "CPU":
        return array.numpy()


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
