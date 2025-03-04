# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("to_tensorflow",)

np = NumpyMetadata.instance()


@high_level_function()
def to_tensorflow(array):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array,
            or low-level #ak.contents.ListOffsetArray, #ak.contents.ListArray,
            #ak.contents.RegularArray, #ak.contents.NumpyArray

    Converts `array` (only ListOffsetArray, ListArray, RegularArray and NumpyArray data types supported)
    into a TensorFlow Tensor, if possible.

    If `array` contains any other data types (RecordArray for example) the function raises a TypeError.
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
            """to use ak.to_tensorflow, you must install the 'tensorflow' package with:

        pip install tensorflow
or
        conda install tensorflow"""
        ) from err

    # useful function that handles all possible input arrays
    array = ak.to_layout(array, allow_record=False)

    # get the device array is on
    ak_device = ak.backend(array)

    if ak_device not in ["cuda", "cpu"]:
        raise ValueError("""Only 'cpu' and 'cuda' backend conversions are allowed""")

    # convert to numpy or cupy if `array` on gpu
    try:
        backend_array = array.to_backend_array(allow_missing=False)
    except ValueError as err:
        raise TypeError(
            "Only arrays containing equal-length lists of numbers can be converted into a TensorFlow Tensor"
        ) from err

    if ak_device == "cpu":
        device = "CPU:0"
    else:
        id = backend_array.data.device.id
        device = "GPU:" + str(id)

    with tf.device(device):
        # check if cupy or numpy
        if isinstance(backend_array, np.ndarray):
            # convert numpy to a tensorflow tensor
            tensor = tf.convert_to_tensor(backend_array, dtype=tf.float64)
        else:
            # cupy -> tensorflow tensor
            tensor = tf.experimental.dlpack.from_dlpack(backend_array.toDlpack())

        return tensor
