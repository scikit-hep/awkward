# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import distutils.version

# NumPy 1.13.1 introduced NEP13, without which Awkward ufuncs won't work,
# which would be worse than lacking a feature: it would cause unexpected output.
# NumPy 1.17.0 introduced NEP18, which is optional (use ak.* instead of np.*).
import numpy
if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion("1.13.1"):
    raise ImportError("Numpy 1.13.1 or later required")

# C++ modules
import awkward1.layout
import awkward1.types

# high-level interface
behavior = {}
from awkward1.highlevel import Array
from awkward1.highlevel import Record
from awkward1.highlevel import FillableArray

# operations and behaviors
from awkward1.operations.convert import *
from awkward1.operations.describe import *
from awkward1.operations.structure import *
from awkward1.operations.reducers import *
from awkward1.behaviors.string import *

# third-party connectors
from awkward1._numexpr import evaluate as numexpr
from awkward1._autograd import elementwise_grad as autograd
def loadnumba():
    print("wowie")
    # called by an entrypoint in setup.py
    import awkward1._numba

# version
__version__ = awkward1.layout.__version__
