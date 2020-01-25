# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import distutils.version

import numpy
if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion("1.13.1"):
    raise ImportError("Numpy 1.13.1 or later required")

classes = {}
functions = {}

import awkward1.layout
from awkward1.layout import Type
from awkward1.layout import UnknownType
from awkward1.layout import PrimitiveType
from awkward1.layout import ListType
from awkward1.layout import RegularType
from awkward1.layout import RecordType
from awkward1.layout import OptionType
from awkward1.layout import UnionType
from awkward1.layout import ArrayType

import awkward1._numba

import awkward1.highlevel
from awkward1.highlevel import Array
from awkward1.highlevel import Record
from awkward1.highlevel import FillableArray

from awkward1.operations.convert import *
from awkward1.operations.describe import *
from awkward1.operations.structure import *

from awkward1.behavior.string import *

__version__ = awkward1.layout.__version__
