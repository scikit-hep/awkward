# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import re

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_tensorflow",)


@high_level_function()
def from_tensorflow(array):
    """
    Args:
        array: (TensorFlow Tensor):
            Tensor to convert into an Awkward Array.
    Converts a TensorFlow Tensor into an Awkward Array.
    If `array` contains any other data types the function raises an error.
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
            """to use ak.from_tensorflow, you must install the 'tensorflow' package with:

        pip install tensorflow
or
        conda install tensorflow"""
        ) from err

    # check if array is a Tensor
    if not isinstance(array, tf.Tensor):
        raise TypeError(
            """only a TensorFlow Tensor can be converted to Awkward Array"""
        )

    # keep the resulting array on the same device as input tensor
    device = array.backing_device
    matched_device = re.match(".*:(CPU|GPU):[0-9]+", device)

    if matched_device is None:
        raise NotImplementedError(
            f"TensorFlow device has an unexpected format: {device!r}"
        )
    elif matched_device.groups()[0] == "GPU":
        from awkward._nplikes.cupy import Cupy

        cp = Cupy.instance()
        # zero-copy data exchange through DLPack
        cp_array = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(array))
        ak_array = ak.from_cupy(cp_array)

    elif matched_device.groups()[0] == "CPU":
        # this makes a copy unfortunately, since numpy is mutable and TensorFlow tensor is not
        np_array = array.numpy()
        ak_array = ak.from_numpy(np_array)

    return ak_array
