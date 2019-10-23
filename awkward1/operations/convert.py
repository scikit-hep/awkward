# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numbers
import json

import numpy

import awkward1.util
import awkward1.layout

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

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "json", "numpy", "awkward1")]
