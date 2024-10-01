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
    array = ak.to_layout(array, allow_record=False)

    # keep the same device
    ak_device = ak.backend(array)
    if ak_device not in ['cuda', 'cpu']:
        raise ValueError("""Only 'cpu' and 'cuda' backend conversions are allowed""")

    if ak_device == 'cpu':
        device = 'CPU:0'
    else:
        device = 'GPU:0'

    with tf.device(device):
        if isinstance(array, ak.contents.numpyarray.NumpyArray):
            values = array.data
            # handle cupy separately
            if not isinstance(array.data, np.ndarray):
                values = _cupy_to_tensor(values)

            return tf.RaggedTensor.from_row_splits(
            values=values, row_splits=[0, array.__len__()]
            )

        else:
            flat_values, nested_row_splits = _recursive_call(array, ())

            ragged_tensor = tf.RaggedTensor.from_nested_row_splits(flat_values, nested_row_splits)
            print(ragged_tensor[0][0].device)
            return ragged_tensor

def _cupy_to_tensor(cupy):
    # converts cupy directly to tensor,
    # since `tf.RaggedTensor.from_nested_row_splits` can not work with Cupy arrays
    import tensorflow as tf
    return tf.experimental.dlpack.from_dlpack(cupy.toDlpack())

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
        if isinstance(offset, np.ndarray):
            offsets_arr += (offset,)
        else:
            offsets_arr += (_cupy_to_tensor(offset),)

    except AttributeError:
        # at the last iteration form a ragged tensor from the
        # accumulated offsets and flattened values of the array
        data = layout.data
        if isinstance(data, np.ndarray):
            return data, offsets_arr
        else:
            return _cupy_to_tensor(data), offsets_arr
    return _recursive_call(layout.content, offsets_arr)
