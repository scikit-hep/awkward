# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# v2: keep this file, but modify it to only get objects that exist!

# NumPy-like alternatives
import awkward.nplike

# shims for C++ (now everything is compiled into one 'awkward._ext' module)
import awkward.layout
import awkward.types
import awkward.forms
import awkward.partition

# internal
import awkward._v2
import awkward._cpu_kernels
import awkward._libawkward
import awkward._util

# third-party connectors
import awkward._connect._numpy
import awkward._connect._numba
import awkward._connect._numexpr
import awkward._connect._autograd
import awkward.numba

# high-level interface
behavior = {}
from awkward.highlevel import Array
from awkward.highlevel import Record
from awkward.highlevel import ArrayBuilder

# third-party jax connectors
import awkward._connect._jax

# behaviors
from awkward.behaviors.mixins import *
from awkward.behaviors.string import *
from awkward.behaviors.categorical import *

# operations
from awkward.operations.convert import *
from awkward.operations.describe import *
from awkward.operations.structure import *
from awkward.operations.reducers import *

# version
__version__ = awkward._ext.__version__

# call C++ startup function
awkward._ext.startup()

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numpy",)]


def __dir__():
    return __all__
