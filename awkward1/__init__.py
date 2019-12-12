# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

namespace = {}

import awkward1.layout
import awkward1._numba
import awkward1.highlevel
from awkward1.highlevel import Array
from awkward1.highlevel import Record

from awkward1.operations.convert import *
from awkward1.operations.describe import *

from awkward1.behavior.string import *

__version__ = awkward1.layout.__version__
