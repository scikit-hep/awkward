# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# layout classes; functionality that used to be in C++ (in Awkward 1.x)
import awkward.index
import awkward.identifier
import awkward.contents
import awkward.record
import awkward.types
import awkward.forms
import awkward._slicing
import awkward._broadcasting
import awkward._typetracer

# internal
import awkward._util
import awkward._lookup

# third-party connectors
import awkward._connect.numpy
import awkward._connect.numexpr
import awkward.numba

# high-level interface
from awkward.highlevel import Array
from awkward.highlevel import Record
from awkward.highlevel import ArrayBuilder

# behaviors
import awkward.behaviors.categorical
import awkward.behaviors.mixins
import awkward.behaviors.string

# operations
from awkward.operations import *


behavior = {}
behaviors.string.register(behavior)  # noqa: F405 pylint: disable=E0602
behaviors.categorical.register(behavior)  # noqa: F405 pylint: disable=E0602
