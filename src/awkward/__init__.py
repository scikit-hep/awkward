# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# Versioning
from awkward._version import __version__

# NumPy-like alternatives
import awkward._backends
import awkward._nplikes

# layout classes; functionality that used to be in C++ (in Awkward 1.x)
import awkward.index
import awkward.contents
import awkward.record
import awkward.types
import awkward.forms
import awkward._do
import awkward._slicing
import awkward._broadcasting
import awkward._reducers

# internal
import awkward._util
import awkward._errors
import awkward._lookup

# third-party connectors
import awkward._connect.numpy
import awkward._connect.numexpr
import awkward.numba
import awkward.cppyy
import awkward.jax
import awkward.typetracer
import awkward._typetracer  # todo: remove this after "deprecation" period

# high-level interface
from awkward.highlevel import *

# behaviors
import awkward.behaviors.categorical
import awkward.behaviors.string
from awkward.behaviors.mixins import *

# exports
import awkward.builder
import awkward.forth

behavior: dict = {}
awkward.behaviors.string.register(behavior)
awkward.behaviors.categorical.register(behavior)

# operations
from awkward.operations import *

# version
__all__ = [x for x in globals() if not x.startswith("_")]


def __dir__():
    return __all__
