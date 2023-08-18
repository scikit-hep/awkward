# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("to_layout",)
from collections.abc import Iterable

from awkward_cpp.lib import _ext

import awkward as ak
from awkward._backends.typetracer import TypeTracerBackend
from awkward._dispatch import high_level_function
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.typetracer import TypeTracer

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@high_level_function()
def to_layout(array, *, allow_record=True, allow_other=False, regulararray=True):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array, #ak.Record (if `allow_record`),
            #ak.ArrayBuilder, or low-level #ak.contents.Content, #ak.record.Record (if `allow_record`),
            or a supported backend array (NumPy `ndarray`, CuPy `ndarray`,
            JAX DeviceArray), data-less TypeTracer, Arrow object, or an arbitrary Python
            iterable (for #ak.from_iter to convert).
        allow_record (bool): If True, allow #ak.record.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allow_other (bool): If True, allow non-Awkward outputs; otherwise,
            if the output would be another type, raise an error.
        regulararray (bool): Prefer to create #ak.contents.RegularArray nodes for
            regular array objects.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.contents.Content and maybe #ak.record.Record or
    other types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis because #ak.contents.Content and
    #ak.record.Record are lower-level than #ak.Array.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, allow_record, allow_other, regulararray=regulararray)


def _impl(array, allow_record, allow_other, regulararray):
    if isinstance(array, ak.contents.Content):
        return array

    elif isinstance(array, ak.record.Record):
        if not allow_record:
            raise TypeError("ak.Record objects are not allowed in this function")
        else:
            return array

    elif isinstance(array, ak.highlevel.Array):
        return array.layout

    elif isinstance(array, ak.highlevel.Record):
        if not allow_record:
            raise TypeError("ak.Record objects are not allowed in this function")
        else:
            return array.layout

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return array.snapshot().layout

    elif isinstance(array, _ext.ArrayBuilder):
        return array.snapshot()

    elif numpy.is_own_array(array):
        return ak.operations.from_numpy(
            array, regulararray=regulararray, recordarray=True, highlevel=False
        )

    elif Cupy.is_own_array(array):
        return ak.operations.from_cupy(
            array, regulararray=regulararray, highlevel=False
        )

    elif Jax.is_own_array(array):
        return ak.operations.from_jax(array, regulararray=regulararray, highlevel=False)

    elif TypeTracer.is_own_array(array):
        backend = TypeTracerBackend.instance()

        if len(array.shape) == 0:
            array = backend.nplike.reshape(array, (1,))

        if array.dtype.kind in {"S", "U"}:
            raise NotImplementedError(
                "strings are currently not supported for typetracer arrays"
            )

        return ak.contents.NumpyArray(array, parameters=None, backend=backend)

    elif ak._util.in_module(array, "pyarrow"):
        return ak.operations.from_arrow(array, highlevel=False)

    elif hasattr(array, "__dlpack__") and hasattr(array, "__dlpack_device__"):
        return ak.operations.from_dlpack(array, highlevel=False)

    elif isinstance(array, (str, bytes)):
        return ak.operations.from_iter([array], highlevel=False)[0]

    elif isinstance(array, Iterable):
        return _impl(
            ak.operations.from_iter(array, highlevel=False),
            allow_record,
            allow_other,
            regulararray,
        )

    elif not allow_other:
        raise TypeError(
            f"{array} cannot be converted into an Awkward Array, and non-array-like objects are not supported."
        )

    else:
        return array
