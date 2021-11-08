# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward._v2.index  # noqa: F401
import awkward._v2.identifier  # noqa: F401
import awkward._v2.contents  # noqa: F401
import awkward._v2.record  # noqa: F401
import awkward._v2.types  # noqa: F401
import awkward._v2.forms  # noqa: F401
import awkward._v2._slicing  # noqa: F401
import awkward._v2._broadcasting  # noqa: F401
import awkward._v2._typetracer  # noqa: F401

import awkward._v2._util  # noqa: F401
import awkward._v2.operations.io  # noqa: F401
import awkward._v2.operations.convert  # noqa: F401
import awkward._v2.operations.describe  # noqa: F401
import awkward._v2.operations.structure  # noqa: F401
import awkward._v2.operations.reducers  # noqa: F401

behavior = {}
import awkward._v2.highlevel  # noqa: F401, E402
import awkward._v2.behaviors.categorical  # noqa: F401, E402
import awkward._v2.behaviors.mixins  # noqa: F401, E402
import awkward._v2.behaviors.string  # noqa: F401, E402
