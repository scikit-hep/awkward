# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numbers

import numpy

import awkward1.layout

def parameters(array):
    if isinstance(array, (awkward1.highlevel.Array, awkward1.highlevel.Record)):
        return array.layout.parameters

    elif isinstance(array, (awkward1.layout.Content, awkward1.layout.Record)):
        return array.parameters

    elif isinstance(array, awkward1.highlevel.FillableArray):
        return array.snapshot().layout.parameters

    elif isinstance(array, awkward1.layout.FillableArray):
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

    elif isinstance(array, (numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64)):
        return awkward1.types.PrimitiveType(typeof.dtype2primitive[array.dtype.type])

    elif isinstance(array, numpy.generic):
        raise ValueError("cannot describe {0} as a PrimitiveType".format(type(array)))

    elif isinstance(array, (awkward1.highlevel.Array, awkward1.highlevel.Record, awkward1.highlevel.FillableArray)):
        return array.type

    elif isinstance(array, awkward1.layout.Record):
        return array.type

    elif isinstance(array, numpy.ndarray):
        if len(array.shape) == 0:
            return typeof(array.reshape((1,))[0])
        else:
            out = awkward1.types.PrimitiveType(typeof.dtype2primitive[array.dtype.type])
            for x in array.shape[-1:0:-1]:
                out = awkward1.types.RegularType(out, x)
            return awkward1.types.ArrayType(out, array.shape[0])

    elif isinstance(array, awkward1.layout.FillableArray):
        return array.type

    elif isinstance(array, awkward1.layout.Content):
        return array.type

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

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "numpy", "awkward1")]
