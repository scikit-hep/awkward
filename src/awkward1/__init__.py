# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import distutils.version

# NumPy 1.13.1 introduced NEP13, without which Awkward ufuncs won't work, which
# would be worse than lacking a feature: it would cause unexpected output.
# NumPy 1.17.0 introduced NEP18, which is optional (use ak.* instead of np.*).
import numpy

if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion(
    "1.13.1"
):
    raise ImportError("Numpy 1.13.1 or later required")

# NumPy-like alternatives
import awkward1.nplike

# shims for C++ (now everything is compiled into one 'awkward1._ext' module)
import awkward1.layout
import awkward1.types
import awkward1.forms
import awkward1.partition

# third-party connectors
import awkward1._connect._numba

numba = type(awkward1._connect._numba)("numba")
numba.register = awkward1._connect._numba.register

import awkward1._connect._pandas

pandas = type(awkward1._connect._pandas)("pandas")
pandas.register = awkward1._connect._pandas.register
pandas.df = awkward1._connect._pandas.df
pandas.dfs = awkward1._connect._pandas.dfs

import awkward1._connect._numexpr

numexpr = type(awkward1._connect._numexpr)("numexpr")
numexpr.evaluate = awkward1._connect._numexpr.evaluate
numexpr.re_evaluate = awkward1._connect._numexpr.re_evaluate

import awkward1._connect._autograd

autograd = type(awkward1._connect._autograd)("autograd")
autograd.elementwise_grad = awkward1._connect._autograd.elementwise_grad

# high-level interface
behavior = {}
from awkward1.highlevel import Array
from awkward1.highlevel import Record
from awkward1.highlevel import ArrayBuilder

# behaviors
from awkward1.behaviors.mixins import mixin_class, mixin_class_method
import awkward1.behaviors.string

# operations
from awkward1.operations.convert import *
from awkward1.operations.describe import *
from awkward1.operations.structure import *
from awkward1.operations.reducers import *
from awkward1.operations.transfers import *

# version
__version__ = awkward1._ext.__version__

# call C++ startup function
awkward1._ext.startup()

__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_") and x not in ("distutils", "numpy")
]
