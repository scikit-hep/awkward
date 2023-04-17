# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("to_layout",)

from collections.abc import Iterable
from numbers import Number

from awkward_cpp.lib import _ext

import awkward as ak
from awkward import _errors
from awkward._layout import from_arraylib
from awkward._nplikes.dispatch import nplike_of
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()
numpy = Numpy.instance()


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
    with _errors.OperationErrorContext(
        "ak.to_layout",
        {
            "array": array,
            "allow_record": allow_record,
            "allow_other": allow_other,
            "regulararray": regulararray,
        },
    ):
        return _impl(array, allow_record, allow_other, regulararray=regulararray)


def _to_layout_detailed(array, allow_record, allow_other, regulararray):
    if isinstance(array, ak.contents.Content):
        return array, False

    elif isinstance(array, ak.record.Record):
        if not allow_record:
            raise TypeError("ak.Record objects are not allowed in this function")
        else:
            return array, True

    elif isinstance(array, ak.highlevel.Array):
        return array.layout, False

    elif isinstance(array, ak.highlevel.Record):
        if not allow_record:
            raise TypeError("ak.Record objects are not allowed in this function")
        else:
            return array.layout, True

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        return array.snapshot().layout, False

    elif isinstance(array, _ext.ArrayBuilder):
        return array.snapshot(), False

    elif nplike_of(array, default=None) is not None:
        # 0D scalar arrays will be promoted
        return (
            from_arraylib(array, regulararray=regulararray, recordarray=True),
            array.ndim == 0,
        )

    elif isinstance(array, np.generic):
        array = numpy.asarray(array)
        return from_arraylib(array, regulararray=regulararray, recordarray=True), True

    elif ak._util.in_module(array, "pyarrow"):
        return ak.operations.from_arrow(array, highlevel=False), False

    elif isinstance(array, (str, bytes, Number, bool)):
        return ak.operations.from_iter([array], highlevel=False), True

    elif array is None:
        return (
            ak.contents.IndexedOptionArray(
                ak.index.Index64(
                    numpy.full(1, -1, dtype=np.int64),
                    nplike=numpy,
                ),
                ak.contents.EmptyArray(),
            ),
            True,
        )

    elif isinstance(array, Iterable):
        return _to_layout_detailed(
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
        return array, True


def _impl(array, allow_record, allow_other, regulararray):
    return _to_layout_detailed(array, allow_record, allow_other, regulararray)[0]
