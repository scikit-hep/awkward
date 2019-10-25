# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numbers
import json
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1.util
import awkward1.layout

def fromiter(iterable, initial=1024, resize=2.0):
    out = awkward1.layout.FillableArray(initial=initial, resize=resize)

    def recurse(obj):
        if obj is None:
            out.null()
        elif isinstance(obj, (bool, numpy.bool, numpy.bool_)):
            out.boolean(obj)
        elif isinstance(obj, (int, numbers.Integral, numpy.integer)):
            out.integer(obj)
        elif isinstance(obj, (float, numbers.Real, numpy.floating)):
            out.real(obj)
        elif isinstance(obj, bytes) and not isinstance(obj, str):
            obj.decode("utf-8")
            out.string(obj)
        elif isinstance(obj, str):
            out.string(obj)
        elif isinstance(obj, dict):
            out.beginrec()
            for n, x in obj.items():
                if isinstance(n, bytes) and not isinstance(n, str):
                    n = n.decode("utf-8")
                if not isinstance(n, str):
                    raise ValueError("only dicts with string-valued keys can be converted with fromiter")
                out.fieldname(n)
                recurse(x)
            out.endrec()
        elif isinstance(obj, Iterable):
            out.beginlist()
            for x in obj:
                recurse(x)
            out.endlist()
        else:
            raise ValueError("cannot convert type: {} value: {}".format(type(obj), repr(obj)))

    recurse(iterable)
    return out.snapshot()

def tolist(array):
    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return array

    elif isinstance(array, numpy.ndarray):
        return array.tolist()

    elif isinstance(array, awkward1.layout.FillableArray):
        return [tolist(x) for x in array]

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array).tolist()

    elif isinstance(array, awkward1.layout.Content):
        return [tolist(x) for x in array]

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

fromjson = awkward1.layout.fromjson

def tojson(array, *args, **kwargs):
    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return json.dumps(array)

    elif isinstance(array, numpy.ndarray):
        return awkward1.layout.NumpyArray(array).tojson(*args, **kwargs)

    elif isinstance(array, awkward1.layout.FillableArray):
        return array.snapshot().tojson(*args, **kwargs)

    elif isinstance(array, awkward1.layout.Content):
        return array.tojson(*args, **kwargs)

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "json", "Iterable", "numpy", "awkward1")]
