# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# NumPy-like alternatives
import awkward.nplikes

# shims for C++ (now everything is compiled into one 'awkward._ext' module)
import awkward._ext

# Compiled dynamic modules
import awkward._cpu_kernels
import awkward._libawkward

# layout classes; functionality that used to be in C++ (in Awkward 1.x)
import awkward.index
import awkward.contents
import awkward.record
import awkward.types
import awkward.forms
import awkward._slicing
import awkward._broadcasting
import awkward._typetracer

# internal
import awkward._util
import awkward._errors
import awkward._lookup

# third-party connectors
import awkward._connect.numpy
import awkward._connect.numexpr
import awkward.numba
import awkward.jax

# high-level interface
from awkward.highlevel import Array
from awkward.highlevel import Record
from awkward.highlevel import ArrayBuilder

# behaviors
import awkward.behaviors.categorical
import awkward.behaviors.string
from awkward.behaviors.mixins import mixin_class, mixin_class_method

behavior: dict = {}
awkward.behaviors.string.register(behavior)  # noqa: F405
awkward.behaviors.categorical.register(behavior)  # noqa: F405

# operations
from awkward.operations import *

# temporary shim to access v2 under _v2
import awkward._v2

# version
__version__ = awkward._ext.__version__
__all__ = [x for x in globals() if not x.startswith("_")]


def __dir__():
    return __all__
