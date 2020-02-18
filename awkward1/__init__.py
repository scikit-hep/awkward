# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import distutils.version

import numpy
if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion("1.13.1"):
    raise ImportError("Numpy 1.13.1 or later required")

behavior = {}

import awkward1._numba

import awkward1.types

import awkward1.highlevel
from awkward1.highlevel import Array
from awkward1.highlevel import Record
from awkward1.highlevel import FillableArray

from awkward1._numexpr import evaluate as numexpr

from awkward1._autograd import elementwise_grad as autograd

from awkward1.operations.convert import *
from awkward1.operations.describe import *
from awkward1.operations.structure import *
from awkward1.operations.reducers import *

from awkward1.behaviors.string import *

__version__ = awkward1.layout.__version__
