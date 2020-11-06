# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numbers
import warnings

import awkward1.layout
import awkward1.nplike


np = awkward1.nplike.NumpyMetadata.instance()
numpy = awkward1.nplike.Numpy.instance()


def is_valid(array, exception=False):
    """
    Args:
        array (#ak.Array, #ak.Record, #ak.layout.Content, #ak.layout.Record, #ak.ArrayBuilder, #ak.layout.ArrayBuilder):
            Array or record to check.
        exception (bool): If True, validity errors raise exceptions.

    Returns True if there are no errors and False if there is an error.

    Checks for errors in the structure of the array, such as indexes that run
    beyond the length of a node's `content`, etc. Either an error is raised or
    the function returns a boolean.

    See also #ak.validity_error.
    """
    out = validity_error(array, exception=exception)
    return out is None


def validity_error(array, exception=False):
    """
    Args:
        array (#ak.Array, #ak.Record, #ak.layout.Content, #ak.layout.Record, #ak.ArrayBuilder, #ak.layout.ArrayBuilder):
            Array or record to check.
        exception (bool): If True, validity errors raise exceptions.

    Returns None if there are no errors and a str containing the error message
    if there are.

    Checks for errors in the structure of the array, such as indexes that run
    beyond the length of a node's `content`, etc. Either an error is raised or
    a string describing the error is returned.

    See also #ak.is_valid.
    """
    if isinstance(array, (awkward1.highlevel.Array, awkward1.highlevel.Record)):
        return validity_error(array.layout, exception=exception)

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return validity_error(array.snapshot().layout, exception=exception)

    elif isinstance(
        array,
        (
            awkward1.layout.Content,
            awkward1.layout.Record,
            awkward1.partition.PartitionedArray,
        ),
    ):
        out = array.validityerror()
        if out is not None and exception:
            raise ValueError(out)
        else:
            return out

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return validity_error(array.snapshot(), exception=exception)

    else:
        raise TypeError(
            "not an awkward array: {0}".format(repr(array))
            + awkward1._util.exception_suffix(__file__)
        )


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
    if array is None:
        return awkward1.types.UnknownType()

    elif isinstance(array, (bool, np.bool, np.bool_)):
        return awkward1.types.PrimitiveType("bool")

    elif isinstance(array, numbers.Integral):
        return awkward1.types.PrimitiveType("int64")

    elif isinstance(array, numbers.Real):
        return awkward1.types.PrimitiveType("float64")

    elif isinstance(
        array,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
        ),
    ):
        return awkward1.types.PrimitiveType(type.dtype2primitive[array.dtype.type])

    elif isinstance(array, awkward1.highlevel.Array):
        return awkward1._util.highlevel_type(array.layout, array.behavior, True)

    elif isinstance(array, awkward1.highlevel.Record):
        return awkward1._util.highlevel_type(array.layout, array.behavior, False)

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return awkward1._util.highlevel_type(
            array.snapshot().layout, array.behavior, True
        )

    elif isinstance(array, awkward1.layout.Record):
        return array.type(awkward1._util.typestrs(None))

    elif isinstance(array, np.ndarray):
        if len(array.shape) == 0:
            return type(array.reshape((1,))[0])
        else:
            try:
                out = type.dtype2primitive[array.dtype.type]
            except KeyError:
                raise TypeError("numpy array type is unrecognized by awkward: %r" % array.dtype.type)
            out = awkward1.types.PrimitiveType(out)
            for x in array.shape[-1:0:-1]:
                out = awkward1.types.RegularType(out, x)
            return awkward1.types.ArrayType(out, array.shape[0])

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return array.type(awkward1._util.typestrs(None))

    elif isinstance(
        array, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
    ):
        return array.type(awkward1._util.typestrs(None))

    else:
        raise TypeError(
            "unrecognized array type: {0}".format(repr(array))
            + awkward1._util.exception_suffix(__file__)
        )


type.dtype2primitive = {
    np.bool: "bool",
    np.bool_: "bool",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.float32: "float32",
    np.float64: "float64",
}


def parameters(array):
    """
    Extracts parameters from the outermost array node of `array` (many types
    supported, including all Awkward Arrays and Records).

    Parameters are a dict from str to JSON-like objects, usually strings.
    Every #ak.layout.Content node has a different set of parameters. Some
    key names are special, such as `"__record__"` and `"__array__"` that name
    particular records and arrays as capable of supporting special behaviors.

    See #ak.Array and #ak.behavior for a more complete description of
    behaviors.
    """
    if isinstance(array, (awkward1.highlevel.Array, awkward1.highlevel.Record)):
        return array.layout.parameters

    elif isinstance(
        array,
        (
            awkward1.layout.Content,
            awkward1.layout.Record,
            awkward1.partition.PartitionedArray,
        ),
    ):
        return array.parameters

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return array.snapshot().layout.parameters

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return array.snapshot().parameters

    else:
        return {}


def fields(array):
    """
    Extracts record fields or tuple slot numbers from `array` (many types
    supported, including all Awkward Arrays and Records).

    If the array contains nested records, only the outermost record is
    queried. If it contains tuples instead of records, this function outputs
    string representations of integers, such as `"0"`, `"1"`, `"2"`, etc.
    The records or tuples may be within multiple layers of nested lists.

    If the array contains neither tuples nor records, this returns an empty
    list.
    """
    layout = awkward1.operations.convert.to_layout(
        array, allow_record=True, allow_other=False
    )
    return layout.keys()


__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_") and x not in (
        "absolute_import",
        "numbers",
        "numpy",
        "np",
        "awkward1",
    )
]
