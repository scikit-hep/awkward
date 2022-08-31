# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def to_layout(
    array,
    allow_record=True,
    allow_other=False,
    numpytype=(np.number, np.bool_, np.str_, np.bytes_, np.datetime64, np.timedelta64),
):
    """
    Args:
        array: Data to convert into a low-level #ak.layout.Content layout
            or maybe #ak.layout.Record, its record equivalent, or other types.
        allow_record (bool): If True, allow #ak.layout.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allow_other (bool): If True, allow non-Awkward outputs; otherwise,
            if the output would be another type, raise an error.
        numpytype (tuple of NumPy types): Dtypes to allow from NumPy arrays.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.layout.Content and maybe #ak.layout.Record or
    other types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis because #ak.layout.Content and
    #ak.layout.Record are lower-level than #ak.Array.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.to_layout",
        dict(
            array=array,
            allow_record=allow_record,
            allow_other=allow_other,
            numpytype=numpytype,
        ),
    ):
        return _impl(array, allow_record, allow_other, numpytype)


def _impl(array, allow_record, allow_other, numpytype):
    if isinstance(array, ak._v2.contents.Content):
        return array

    elif isinstance(array, ak._v2.record.Record):
        if not allow_record:
            raise ak._v2._util.error(
                TypeError("ak._v2.Record objects are not allowed in this function")
            )
        else:
            return array

    elif isinstance(array, ak._v2.highlevel.Array):
        return array.layout

    elif isinstance(array, ak._v2.highlevel.Record):
        if not allow_record:
            raise ak._v2._util.error(
                TypeError("ak._v2.Record objects are not allowed in this function")
            )
        else:
            return array.layout

    # elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
    #     return array.snapshot().layout

    # elif isinstance(array, ak.layout.ArrayBuilder):
    #     return array.snapshot()

    elif isinstance(array, (np.ndarray, numpy.ma.MaskedArray)):
        if not issubclass(array.dtype.type, numpytype):
            raise ak._v2._util.error(ValueError(f"dtype {array.dtype!r} not allowed"))
        return _impl(
            ak._v2.operations.from_numpy(
                array, regulararray=True, recordarray=True, highlevel=False
            ),
            allow_record,
            allow_other,
            numpytype,
        )

    elif ak.nplike.is_cupy_buffer(array) and type(array).__name__ == "ndarray":
        if not issubclass(array.dtype.type, numpytype):
            raise ak._v2._util.error(ValueError(f"dtype {array.dtype!r} not allowed"))
        return _impl(
            ak._v2.operations.from_cupy(array, regulararray=True, highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif ak.nplike.is_jax_buffer(array) and type(array).__name__ == "DeviceArray":
        if not issubclass(array.dtype.type, numpytype):
            raise ak._v2._util.error(ValueError(f"dtype {array.dtype!r} not allowed"))
        return _impl(
            ak._v2.operations.from_jax(array, regulararray=True, highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif isinstance(array, (str, bytes)):
        return _impl(
            ak._v2.operations.from_iter([array], highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif isinstance(array, Iterable):
        return _impl(
            ak._v2.operations.from_iter(array, highlevel=False),
            allow_record,
            allow_other,
            numpytype,
        )

    elif not allow_other:
        raise ak._v2._util.error(
            TypeError(f"{array} cannot be converted into an Awkward Array")
        )

    else:
        return array
