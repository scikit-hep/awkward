# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable

from awkward_cpp.lib import _ext

import awkward as ak
from awkward import _errors

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


def to_layout(array, *, allow_record=True, allow_other=False):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array, #ak.Record (if `allow_record`),
            #ak.ArrayBuilder, or low-level #ak.contents.Content, #ak.record.Record (if `allow_record`),
            or a supported backend array (NumPy `ndarray`, CuPy `ndarray`,
            JAX DeviceArray), dataless TypeTracer, or an arbitrary Python
            iterable (for #ak.from_iter to convert).
        allow_record (bool): If True, allow #ak.record.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allow_other (bool): If True, allow non-Awkward outputs; otherwise,
            if the output would be another type, raise an error.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.contents.Content and maybe #ak.record.Record or
    other types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis because #ak.contents.Content and
    #ak.record.Record are lower-level than #ak.Array.
    """
    with _errors.OperationErrorContext(
        "ak.to_layout",
        dict(array=array, allow_record=allow_record, allow_other=allow_other),
    ):
        return _impl(array, allow_record, allow_other)


def _impl(array, allow_record, allow_other):
    if isinstance(array, ak.contents.Content):
        return array

    elif isinstance(array, ak.record.Record):
        if not allow_record:
            raise _errors.wrap_error(
                TypeError("ak.Record objects are not allowed in this function")
            )
        else:
            return array

    elif isinstance(array, ak.highlevel.Array):
        return array.layout

    elif isinstance(array, ak.highlevel.Record):
        if not allow_record:
            raise _errors.wrap_error(
                TypeError("ak.Record objects are not allowed in this function")
            )
        else:
            return array.layout

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return array.snapshot().layout

    elif isinstance(array, _ext.ArrayBuilder):
        return array.snapshot()

    elif numpy.is_own_array(array):
        return _impl(
            ak.operations.from_numpy(
                array, regulararray=True, recordarray=True, highlevel=False
            ),
            allow_record,
            allow_other,
        )

    elif ak._nplikes.Cupy.is_own_array(array):
        return _impl(
            ak.operations.from_cupy(array, regulararray=True, highlevel=False),
            allow_record,
            allow_other,
        )

    elif ak._nplikes.Jax.is_own_array(array):
        return _impl(
            ak.operations.from_jax(array, regulararray=True, highlevel=False),
            allow_record,
            allow_other,
        )

    elif ak._typetracer.TypeTracer.is_own_array(array):
        backend = ak._backends.TypeTracerBackend.instance()

        if len(array.shape) == 0:
            array = array.reshape(1)

        if array.dtype.kind in {"S", "U"}:
            raise _errors.wrap_error(
                NotImplementedError(
                    "strings are currently not supported for typetracer arrays"
                )
            )

        return ak.contents.NumpyArray(array, parameters=None, backend=backend)

    elif isinstance(array, (str, bytes)):
        return _impl(
            ak.operations.from_iter([array], highlevel=False),
            allow_record,
            allow_other,
        )

    elif isinstance(array, Iterable):
        return _impl(
            ak.operations.from_iter(array, highlevel=False),
            allow_record,
            allow_other,
        )

    elif not allow_other:
        raise _errors.wrap_error(
            TypeError(f"{array} cannot be converted into an Awkward Array")
        )

    else:
        return array
