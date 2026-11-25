# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


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
    primitive_policy="error",
    attrs=None,
):
    """Converts a DLPack-aware array into an Awkward Array.

    The data is not copied: the buffer is shared through the DLPack protocol.

    The resulting layout may involve the following #ak.contents.Content types
    (only):

    * #ak.contents.NumpyArray
    * #ak.contents.RegularArray if `regulararray=True`.

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

    Returns:
        An #ak.Array built from the given DLPack-aware array.
    """
    try:
        dlpack_info_func = array.__dlpack_device__
    except AttributeError as err:
        raise TypeError(
            f"Expected an object that implements the DLPack protocol, received {type(array)}"
        ) from err
    device_type, _device_id = dlpack_info_func()

    # Only a subset of known devices are supported.
    nplike: NumpyLike
    match device_type:
        # TODO: CUDA_PINNED should support GPU
        case DLPackDevice.CPU | DLPackDevice.CUDA_PINNED:
            nplike = Numpy.instance()
        case DLPackDevice.CUDA | DLPackDevice.ROCM:
            nplike = Cupy.instance()
        case DLPackDevice.ROCM_PINNED | DLPackDevice.CUDA_MANAGED:
            nplike = (Numpy if prefer_cpu else Cupy).instance()
        case _:
            raise NotImplementedError(
                f"DLPack device type {DLPackDevice(device_type).name} is not supported"
            )

    array = nplike.from_dlpack(array)
    return wrap_layout(
        from_arraylib(array, regulararray, False, primitive_policy=primitive_policy),
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
