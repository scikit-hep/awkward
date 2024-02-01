# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# Versioning
from __future__ import annotations
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

# internal
import awkward._do
import awkward._slicing
import awkward._broadcasting
import awkward._reducers
import awkward._util
import awkward._errors
import awkward._lookup
import awkward._ext  # strictly for unpickling from Awkward 1

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
from awkward.behaviors.mixins import *

# exports
import awkward.builder
import awkward.forth

# errors
import awkward.errors

behavior: dict = {}

# operations
from awkward.operations import *

# version
__all__ = [x for x in globals() if not x.startswith("_")]


def __dir__():
    return __all__
