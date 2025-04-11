# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

# Versioning
from awkward._version import __version__

# NumPy-like alternatives
from awkward import _backends
from awkward import _nplikes

# layout classes; functionality that used to be in C++ (in Awkward 1.x)
from awkward import index
from awkward import contents
from awkward import record
from awkward import types
from awkward import forms

# internal
from awkward import _do
from awkward import _slicing
from awkward import _broadcasting
from awkward import _reducers
from awkward import _util
from awkward import _errors
from awkward import _lookup
from awkward import _ext  # strictly for unpickling from Awkward 1
from awkward import _namedaxis

# third-party connectors
from awkward._connect import numpy
from awkward._connect import numexpr
from awkward import numba
from awkward import cppyy
from awkward import jax
from awkward import typetracer
from awkward import _typetracer  # todo: remove this after "deprecation" period

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
