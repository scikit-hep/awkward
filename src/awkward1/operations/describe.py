# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numbers

import numpy

import awkward1.layout

def keys(array):
    """
    Extracts record or tuple keys from `array` (many types supported,
    including all Awkward Arrays and Records).

    If the array contains nested records, only the outermost record is
    queried. If it contains tuples instead of records, the keys are
    string representations of integers, such as `"0"`, `"1"`, `"2"`, etc.
    The records or tuples may be within multiple layers of nested lists.

    If the array contains neither tuples nor records, this returns an empty
    list.
    """
    layout = awkward1.operations.convert.tolayout(array,
                                                  allowrecord=True,
                                                  allowother=False)
    return layout.keys()

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
    if isinstance(array, (awkward1.highlevel.Array,
                          awkward1.highlevel.Record)):
        return array.layout.parameters

    elif isinstance(array, (awkward1.layout.Content,
                            awkward1.layout.Record)):
        return array.parameters

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return array.snapshot().layout.parameters

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return array.snapshot().parameters

    else:
        return {}

def typeof(array):
    if array is None:
        return awkward1.types.UnknownType()

    elif isinstance(array, (bool, numpy.bool, numpy.bool_)):
        return awkward1.types.PrimitiveType("bool")

    elif isinstance(array, numbers.Integral):
        return awkward1.types.PrimitiveType("int64")

    elif isinstance(array, numbers.Real):
        return awkward1.types.PrimitiveType("float64")

    elif isinstance(array,
                    (numpy.int8,
                     numpy.int16,
                     numpy.int32,
                     numpy.int64,
                     numpy.uint8,
                     numpy.uint16,
                     numpy.uint32,
                     numpy.uint64,
                     numpy.float32,
                     numpy.float64)):
        return awkward1.types.PrimitiveType(
                 typeof.dtype2primitive[array.dtype.type])

    elif isinstance(array, (awkward1.highlevel.Array,
                            awkward1.highlevel.Record,
                            awkward1.highlevel.ArrayBuilder)):
        return array.type

    elif isinstance(array, awkward1.layout.Record):
        return array.type(awkward1._util.typestrs(None))

    elif isinstance(array, numpy.ndarray):
        if len(array.shape) == 0:
            return typeof(array.reshape((1,))[0])
        else:
            out = awkward1.types.PrimitiveType(
                    typeof.dtype2primitive[array.dtype.type])
            for x in array.shape[-1:0:-1]:
                out = awkward1.types.RegularType(out, x)
            return awkward1.types.ArrayType(out, array.shape[0])

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return array.type(awkward1._util.typestrs(None))

    elif isinstance(array, awkward1.layout.Content):
        return array.type(awkward1._util.typestrs(None))

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

typeof.dtype2primitive = {
    numpy.int8:    "int8",
    numpy.int16:   "int16",
    numpy.int32:   "int32",
    numpy.int64:   "int64",
    numpy.uint8:   "uint8",
    numpy.uint16:  "uint16",
    numpy.uint32:  "uint32",
    numpy.uint64:  "uint64",
    numpy.float32: "float32",
    numpy.float64: "float64",
}

def validityerror(array, exception=False):
    if isinstance(array, (awkward1.highlevel.Array,
                          awkward1.highlevel.Record)):
        return validityerror(array.layout, exception=exception)

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return validityerror(array.snapshot().layout, exception=exception)

    elif isinstance(array, (awkward1.layout.Content, awkward1.layout.Record)):
        out = array.validityerror()
        if out is not None and exception:
            raise ValueError(out)
        else:
            return out

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return validityerror(array.snapshot(), exception=exception)

    else:
        raise TypeError("not an awkward array: {0}".format(repr(array)))

def isvalid(array, exception=False):
    out = validityerror(array, exception=exception)
    return out is None

__all__ = [x for x in list(globals())
             if not x.startswith("_") and
             x not in ("numbers", "numpy", "awkward1")]
