# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("to_raggedtensor",)

np = NumpyMetadata.instance()


@high_level_function()
def to_raggedtensor(array):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array,
            or low-level #ak.contents.ListOffsetArray, #ak.contents.ListArray,
            #ak.contents.RegularArray, #ak.contents.NumpyArray

    Converts `array` (only ListOffsetArray, ListArray, RegularArray and NumpyArray data types supported)
    into a ragged tensor, if possible.

    If `array` contains any other data types (RecordArray for example) the function raises an error.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import tensorflow as tf
    except ImportError as err:
        raise ImportError(
            """to use ak.to_raggedtensor, you must install the 'tensorflow' package with:

        pip install tensorflow
or
        conda install tensorflow"""
        ) from err

    # unwrap the awkward array if it was made with ak.Array function
    # also transforms a python list to awkward array
    array = ak.to_layout(
        ak.operations.materialize(array)
        if isinstance(array, (ak.highlevel.Array, ak.contents.Content))
        else array,
        allow_record=False,
    )

    # keep the same device
    ak_device = ak.backend(array)
    if ak_device not in ["cuda", "cpu"]:
        raise ValueError("""Only 'cpu' and 'cuda' backend conversions are allowed""")

    if ak_device == "cpu":
        device = "CPU:0"
    else:
        id = _find_innermost_content(array).data.device.id
        device = "GPU:" + str(id)

    with tf.device(device):
        if isinstance(array, ak.contents.numpyarray.NumpyArray):
            values = array.data
            # handle cupy separately
            values = _convert_to_tensor_if_cupy(values)
            return tf.RaggedTensor.from_row_splits(
                values=values, row_splits=[0, array.__len__()]
            )

        else:
            flat_values, nested_row_splits = _recursive_call(array, ())
            return tf.RaggedTensor.from_nested_row_splits(
                flat_values, nested_row_splits
            )


def _find_innermost_content(array):
    if isinstance(array, ak.contents.numpyarray.NumpyArray):
        return array
    else:
        return _find_innermost_content(array.content)


def _convert_to_tensor_if_cupy(array):
    if isinstance(array, np.ndarray):
        return array
    else:
        # converts cupy directly to tensor,
        # since `tf.RaggedTensor.from_nested_row_splits` can not work with Cupy arrays
        import tensorflow as tf

        return tf.experimental.dlpack.from_dlpack(array.toDlpack())


def _recursive_call(layout, offsets_arr):
    try:
        # change all the possible layout types to ListOffsetArray
        if isinstance(layout, ak.contents.listarray.ListArray):
            layout = layout.to_ListOffsetArray64()
        elif isinstance(layout, ak.contents.regulararray.RegularArray):
            layout = layout.to_ListOffsetArray64()
        elif not isinstance(
            layout,
            (
                ak.contents.listoffsetarray.ListOffsetArray,
                ak.contents.numpyarray.NumpyArray,
            ),
        ):
            raise TypeError(
                "Only arrays containing variable-length lists (var *) or"
                " regular-length lists (# *) of numbers can be converted into a TensorFlow RaggedTensor"
            )

        # recursively gather all of the offsets of an array
        offset = layout.offsets.data
        offset = _convert_to_tensor_if_cupy(offset)
        offsets_arr += (offset,)

    except AttributeError:
        # at the last iteration form a ragged tensor from the
        # accumulated offsets and flattened values of the array
        data = layout.data
        data = _convert_to_tensor_if_cupy(data)
        return data, offsets_arr
    return _recursive_call(layout.content, offsets_arr)
