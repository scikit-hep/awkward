# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

# layout classes; functionality that used to be in C++ (in Awkward 1.x)
import awkward._v2.index  # noqa: F401
import awkward._v2.identifier  # noqa: F401
import awkward._v2.contents  # noqa: F401
import awkward._v2.record  # noqa: F401
import awkward._v2.types  # noqa: F401
import awkward._v2.forms  # noqa: F401
import awkward._v2._slicing  # noqa: F401
import awkward._v2._broadcasting  # noqa: F401
import awkward._v2._typetracer  # noqa: F401

# internal
import awkward._v2._util  # noqa: F401

# third-party connectors
import awkward._v2._connect.numpy
import awkward._v2._connect.numba
import awkward._v2._connect.numexpr  # noqa: F401

# high-level interface
from awkward._v2.highlevel import Array  # noqa: F401
from awkward._v2.highlevel import Record  # noqa: F401
from awkward._v2.highlevel import ArrayBuilder  # noqa: F401

# behaviors
behavior = {}
from awkward._v2.behaviors.categorical import *  # noqa: E402, F401, F403
from awkward._v2.behaviors.mixins import *  # noqa: E402, F401, F403
from awkward._v2.behaviors.string import *  # noqa: E402, F401, F403

# operations
from awkward._v2.operations.io import *  # noqa: E402, F401, F403
from awkward._v2.operations.convert import *  # noqa: E402, F401, F403
from awkward._v2.operations.describe import *  # noqa: E402, F401, F403
from awkward._v2.operations.structure import *  # noqa: E402, F401, F403
from awkward._v2.operations.reducers import *  # noqa: E402, F401, F403
