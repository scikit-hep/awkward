# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

# Versioning
from awkward._version import __version__

# NumPy-like alternatives
import awkward._backends
import awkward._nplikes

# layout classes; functionality that used to be in C++ (in Awkward 1.x)
from awkward import index
from awkward import contents
from awkward import record
from awkward import types
from awkward import forms

# internal
import awkward._do
import awkward._slicing
import awkward._broadcasting
import awkward._reducers
import awkward._util
import awkward._errors
import awkward._lookup
import awkward._ext  # strictly for unpickling from Awkward 1
import awkward._namedaxis

# third-party connectors
from awkward._connect import numpy
from awkward._connect import numexpr
from awkward import numba
from awkward import cppyy
from awkward import jax
from awkward import typetracer
import awkward._typetracer  # todo: remove this after "deprecation" period

# high-level interface
from awkward.highlevel import *

# behaviors
from awkward.behaviors.mixins import *

# exports
from awkward import builder
from awkward import forth

# errors
from awkward import errors

behavior: dict = {}

# operations
from awkward.operations import *

# version
__all__ = [x for x in globals() if not x.startswith("_")]


def __dir__():
    return __all__
