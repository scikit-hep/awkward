# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

import numpy

from awkward._nplikes.array_module import ArrayModuleNumpyLike


class Numpy(ArrayModuleNumpyLike):
    """
    A concrete class importing `NumpyModuleLike` for `numpy`
    """

    array_module = numpy
