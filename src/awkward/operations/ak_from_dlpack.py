# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._connect.dlpack import DLPackDevice
from awkward._dispatch import high_level_function
from awkward._layout import from_arraylib, wrap_layout
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyLike

__all__ = ("from_dlpack",)


@high_level_function()
def from_dlpack(
    array,
    *,
    prefer_cpu=True,
    regulararray=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array (cp.ndarray): The DLPack-supporting array to convert into an
            Awkward Array.
        prefer_cpu (bool): If True, and the array device supports both CPU and
            GPU backends, prefer the CPU; otherwise, prefer the GPU.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.contents.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.contents.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts a DLPack-aware array into an Awkward Array.

    The resulting layout may involve the following #ak.contents.Content types
    (only):

    * #ak.contents.NumpyArray
    * #ak.contents.RegularArray if `regulararray=True`.
    """
    try:
        dlpack_info_func = array.__dlpack_device__
    except AttributeError as err:
        raise TypeError(
            f"Expected an object that implements the DLPack protocol, received {type(array)}"
        ) from err
    device_type, device_id = dlpack_info_func()

    # Only a subset of known devices are supported.
    nplike: NumpyLike
    if device_type == DLPackDevice.CPU:
        nplike = Numpy.instance()
    elif device_type == DLPackDevice.CUDA:
        nplike = Cupy.instance()
    elif device_type == DLPackDevice.CUDA_PINNED:
        # TODO: this should support GPU
        # nplike = (Numpy if prefer_cpu else Cupy).instance()
        nplike = Numpy.instance()
    elif device_type == DLPackDevice.ROCM:
        nplike = Cupy.instance()
    elif device_type == DLPackDevice.ROCM_PINNED:
        nplike = (Numpy if prefer_cpu else Cupy).instance()
    elif device_type == DLPackDevice.CUDA_MANAGED:
        nplike = (Numpy if prefer_cpu else Cupy).instance()
    else:
        raise AssertionError

    array = nplike.from_dlpack(array)
    return wrap_layout(
        from_arraylib(array, regulararray, False),
        highlevel=highlevel,
        behavior=behavior,
    )
