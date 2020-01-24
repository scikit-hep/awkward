# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import numbers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1._util
import awkward1.layout
import awkward1._npfunctions
import awkward1.operations.convert

@awkward1._npfunctions.implements(numpy.where)
def where(condition, *args):
    if len(args) == 0:
        raise NotImplementedError("ak.where(condition)")

    elif len(args) == 1:
        raise ValueError("either both or neither of x and y should be given")

    elif len(args) == 2:
        x, y = args
        raise NotImplementedError("ak.where(condition, x, y)")

    else:
        raise TypeError("where() takes from 1 to 3 positional arguments but {0} were given".format(len(args) + 1))

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "Iterable", "numpy", "awkward1")]
