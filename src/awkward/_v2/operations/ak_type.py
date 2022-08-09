# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numbers

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def type(array):
    """
    The high-level type of an `array` (many types supported, including all
    Awkward Arrays and Records) as #ak.types.Type objects.

    The high-level type ignores #layout differences like
    #ak.layout.ListArray64 versus #ak.layout.ListOffsetArray64, but
    not differences like "regular-sized lists" (i.e.
    #ak.layout.RegularArray) versus "variable-sized lists" (i.e.
    #ak.layout.ListArray64 and similar).

    Types are rendered as [Datashape](https://datashape.readthedocs.io/)
    strings, which makes the same distinctions.

    For example,

        ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
                  [],
                  [{"x": 3.3, "y": [3, 3, 3]}]])

    has type

        3 * var * {"x": float64, "y": var * int64}

    but

        ak.Array(np.arange(2*3*5).reshape(2, 3, 5))

    has type

        2 * 3 * 5 * int64

    Some cases, like heterogeneous data, require [extensions beyond the
    Datashape specification](https://github.com/blaze/datashape/issues/237).
    For example,

        ak.Array([1, "two", [3, 3, 3]])

    has type

        3 * union[int64, string, var * int64]

    but "union" is not a Datashape type-constructor. (Its syntax is
    similar to existing type-constructors, so it's a plausible addition
    to the language.)
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.type",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    if array is None:
        return ak._v2.types.UnknownType()

    elif isinstance(
        array,
        tuple(x.type for x in ak._v2.types.numpytype._dtype_to_primitive_dict),
    ):
        return ak._v2.types.NumpyType(
            ak._v2.types.numpytype._dtype_to_primitive_dict[array.dtype]
        )

    elif isinstance(array, (bool, np.bool_)):
        return ak._v2.types.NumpyType("bool")

    elif isinstance(array, numbers.Integral):
        return ak._v2.types.NumpyType("int64")

    elif isinstance(array, numbers.Real):
        return ak._v2.types.NumpyType("float64")

    elif isinstance(
        array,
        (
            ak._v2.highlevel.Array,
            ak._v2.highlevel.Record,
            ak._v2.highlevel.ArrayBuilder,
        ),
    ):
        return array.type

    elif isinstance(array, np.ndarray):
        if len(array.shape) == 0:
            return _impl(array.reshape((1,))[0])
        else:
            try:
                out = ak._v2.types.numpytype._dtype_to_primitive_dict[array.dtype.type]
            except KeyError as err:
                raise ak._v2._util.error(
                    TypeError(
                        "numpy array type is unrecognized by awkward: %r"
                        % array.dtype.type
                    )
                ) from err
            out = ak._v2.types.NumpyType(out)
            for x in array.shape[-1:0:-1]:
                out = ak._v2.types.RegularType(out, x)
            return ak._v2.types.ArrayType(out, array.shape[0])

    elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
        return NotImplementedError

    elif isinstance(array, ak._v2.record.Record):
        return array.array.form.type

    elif isinstance(array, ak._v2.contents.Content):
        return array.form.type

    elif isinstance(
        array,
        (
            ak.highlevel.Array,
            ak.highlevel.Record,
            ak.highlevel.ArrayBuilder,
            ak.layout.Content,
            ak.layout.Record,
        ),
    ):
        raise ak._v2._util.error(
            TypeError("do not use ak._v2.operations.type on v1 arrays")
        )

    else:
        raise ak._v2._util.error(TypeError(f"unrecognized array type: {array!r}"))
