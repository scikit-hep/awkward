# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable

from awkward_cpp.lib import _ext

import awkward as ak
from awkward import _errors

np = ak.nplikes.NumpyMetadata.instance()
numpy = ak.nplikes.Numpy.instance()


def to_layout(
    array,
    allow_record=True,
    allow_other=False,
    numpytype=(np.number, np.bool_, np.str_, np.bytes_, np.datetime64, np.timedelta64),
):
    """
    Args:
        array: Data to convert into a low-level #ak.contents.Content layout
            or maybe #ak.record.Record, its record equivalent, or other types.
        allow_record (bool): If True, allow #ak.record.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allow_other (bool): If True, allow non-Awkward outputs; otherwise,
            if the output would be another type, raise an error.
        numpytype (tuple of NumPy types): Dtypes to allow from NumPy arrays.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.contents.Content and maybe #ak.record.Record or
    other types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis because #ak.contents.Content and
    #ak.record.Record are lower-level than #ak.Array.
    """
    with _errors.OperationErrorContext(
        "ak.to_layout",
        dict(
            array=array,
            allow_record=allow_record,
            allow_other=allow_other,
            numpytype=numpytype,
        ),
    ):
        return _impl(array, allow_record, allow_other, numpytype)


def _impl(array, allow_record, allow_other, numpytype):
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
        if not issubclass(array.dtype.type, numpytype):
            raise _errors.wrap_error(ValueError(f"dtype {array.dtype!r} not allowed"))
        return _impl(
            ak.operations.from_numpy(
                array, regulararray=True, recordarray=True, highlevel=False
            ),
            allow_record,
            allow_other,
            numpytype,
        )

    elif ak.nplikes.Cupy.is_own_array(array):
        if not issubclass(array.dtype.type, numpytype):
            raise _errors.wrap_error(ValueError(f"dtype {array.dtype!r} not allowed"))
        return _impl(
            ak.operations.from_cupy(array, regulararray=True, highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif ak.nplikes.Jax.is_own_array(array):
        if not issubclass(array.dtype.type, numpytype):
            raise _errors.wrap_error(ValueError(f"dtype {array.dtype!r} not allowed"))
        return _impl(
            ak.operations.from_jax(array, regulararray=True, highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif ak._typetracer.TypeTracer.is_own_array(array):
        typetracer = ak._typetracer.TypeTracer.instance()

        if len(array.shape) == 0:
            array = array.reshape(1)

        if array.dtype.kind in {"S", "U"}:
            raise _errors.wrap_error(
                NotImplementedError(
                    "strings are currently not supported for typetracer arrays"
                )
            )

        return ak.contents.NumpyArray(array, parameters=None, nplike=typetracer)

    elif isinstance(array, (str, bytes)):
        return _impl(
            ak.operations.from_iter([array], highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif isinstance(array, Iterable):
        return _impl(
            ak.operations.from_iter(array, highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif not allow_other:
        raise _errors.wrap_error(
            TypeError(f"{array} cannot be converted into an Awkward Array")
        )

    else:
        return array
