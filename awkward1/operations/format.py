# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy

import awkward1.layout

def tolist(array):
    if isinstance(array, numpy.ndarray):
        return array.tolist()

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array).tolist()

    elif isinstance(array, (awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArray64)):
        return [tolist(x) for x in array]

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("awkward1", "numpy")]
