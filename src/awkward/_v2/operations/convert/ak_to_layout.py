# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

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
    if isinstance(array, ak._v2.contents.Content):
        return array

    elif allow_record and isinstance(array, ak._v2.record.Record):
        return array

    elif isinstance(array, ak._v2.highlevel.Array):
        return array.layout

    elif allow_record and isinstance(array, ak._v2.highlevel.Record):
        return array.layout

    # elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
    #     return array.snapshot().layout

    # elif isinstance(array, ak.layout.ArrayBuilder):
    #     return array.snapshot()

    elif isinstance(array, (np.ndarray, numpy.ma.MaskedArray)):
        if not issubclass(array.dtype.type, numpytype):
            raise ValueError("dtype {0} not allowed".format(repr(array.dtype)))
        return to_layout(
            ak._v2.operations.convert.from_numpy(
                array, regulararray=True, recordarray=True, highlevel=False
            ),
            allow_record=allow_record,
            allow_other=allow_other,
        )

    elif (
        type(array).__module__.startswith("cupy.") and type(array).__name__ == "ndarray"
    ):
        if not issubclass(array.dtype.type, numpytype):
            raise ValueError("dtype {0} not allowed".format(repr(array.dtype)))
        return to_layout(
            ak._v2.operations.convert.from_cupy(
                array, regulararray=True, highlevel=False
            ),
            allow_record=allow_record,
            allow_other=allow_other,
        )

    elif isinstance(array, (str, bytes)):
        return to_layout(
            ak._v2.operations.convert.from_iter([array], highlevel=False),
            allow_record=allow_record,
            allow_other=allow_other,
        )

    elif isinstance(array, Iterable):
        return to_layout(
            ak._v2.operations.convert.from_iter(array, highlevel=False),
            allow_record=allow_record,
            allow_other=allow_other,
        )

    elif not allow_other:
        raise TypeError("{0} cannot be converted into an Awkward Array".format(array))

    else:
        return array
