# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import awkward1.layout
import awkward1._numba
import awkward1.highlevel

from awkward1.operations.convert import *
from awkward1.operations.describe import *

class Array(awkward1.highlevel.Array):
    pass

class Record(awkward1.highlevel.Record):
    pass

__version__ = awkward1.layout.__version__
