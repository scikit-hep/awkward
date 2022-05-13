# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# layout classes; functionality that used to be in C++ (in Awkward 1.x)
import awkward._v2.index
import awkward._v2.identifier
import awkward._v2.contents
import awkward._v2.record
import awkward._v2.types
import awkward._v2.forms
import awkward._v2._slicing
import awkward._v2._broadcasting
import awkward._v2._typetracer

# internal
import awkward._v2._util
import awkward._v2._lookup

# third-party connectors
import awkward._v2._connect.numpy
import awkward._v2._connect.numexpr
import awkward._v2.numba

# high-level interface
from awkward._v2.highlevel import Array
from awkward._v2.highlevel import Record
from awkward._v2.highlevel import ArrayBuilder

# behaviors
import awkward._v2.behaviors.categorical
import awkward._v2.behaviors.mixins
import awkward._v2.behaviors.string

# operations
from awkward._v2.operations import *


behavior = {}
behaviors.string.register(behavior)  # noqa: F405 pylint: disable=E0602
behaviors.categorical.register(behavior)  # noqa: F405 pylint: disable=E0602
