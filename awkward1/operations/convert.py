# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import numbers
import json
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1._util
import awkward1.layout

def fromnumpy(array):
    def recurse(array):
        if len(array.shape) == 0:
            return awkward1.layout.NumpyArray(array.reshape(1))
        elif len(array.shape) == 1:
            return awkward1.layout.NumpyArray(array)
        else:
            return awkward1.layout.RegularArray(recurse(array.reshape((-1,) + array.shape[2:])), array.shape[1])
    return awkward1._util.wrap(recurse(array), awkward1.classes, awkward1.functions)

def fromiter(iterable, initial=1024, resize=2.0):
    out = awkward1.layout.FillableArray(initial=initial, resize=resize)
    for x in iterable:
        out.fill(x)
    return awkward1._util.wrap(out.snapshot(), awkward1.classes, awkward1.functions)

def fromjson(source, initial=1024, resize=2.0, buffersize=65536):
    return awkward1._util.wrap(awkward1.layout.fromjson(source, initial=initial, resize=resize, buffersize=buffersize), awkward1.classes, awkward1.functions)

def tolist(array):
    import awkward1.highlevel

    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return array

    elif sys.version_info[0] < 3 and isinstance(array, unicode):
        return array

    elif isinstance(array, awkward1.behavior.string.CharBehavior):
        if array.layout.parameters.get("encoding") is None:
            return array.__bytes__()
        else:
            return array.__str__()

    elif awkward1.operations.describe.parameters(array).get("__class__") == "char":
        if awkward1.operations.describe.parameters(array).get("encoding") is None:
            return awkward1.behavior.string.CharBehavior(array).__bytes__()
        else:
            return awkward1.behavior.string.CharBehavior(array).__str__()

    elif isinstance(array, awkward1.highlevel.Array):
        return [tolist(x) for x in array]

    elif isinstance(array, awkward1.highlevel.Record):
        return tolist(array.layout)

    elif isinstance(array, awkward1.highlevel.FillableArray):
        return tolist(array.snapshot())

    elif isinstance(array, awkward1.layout.Record) and array.istuple:
        return tuple(tolist(x) for x in array.fields())

    elif isinstance(array, awkward1.layout.Record):
        return {n: tolist(x) for n, x in array.fielditems()}

    elif isinstance(array, numpy.ndarray):
        return array.tolist()

    elif isinstance(array, awkward1.layout.FillableArray):
        return [tolist(x) for x in array.snapshot()]

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array).tolist()

    elif isinstance(array, awkward1.layout.Content):
        return [tolist(x) for x in array]

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

def tojson(array, destination=None, pretty=False, maxdecimals=None, buffersize=65536):
    import awkward1.highlevel

    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return json.dumps(array)

    elif isinstance(array, awkward1.highlevel.Array):
        out = array.layout

    elif isinstance(array, awkward1.highlevel.Record):
        out = array.layout

    elif isinstance(array, awkward1.highlevel.FillableArray):
        out = array.snapshot().layout

    elif isinstance(array, awkward1.layout.Record):
        out = array

    elif isinstance(array, numpy.ndarray):
        out = awkward1.layout.NumpyArray(array)

    elif isinstance(array, awkward1.layout.FillableArray):
        out = array.snapshot()

    elif isinstance(array, awkward1.layout.Content):
        out = array

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

    if destination is None:
        return out.tojson(pretty=pretty, maxdecimals=maxdecimals)
    else:
        return out.tojson(destination, pretty=pretty, maxdecimals=maxdecimals, buffersize=buffersize)

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "json", "Iterable", "numpy", "awkward1")]
