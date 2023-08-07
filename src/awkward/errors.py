# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("FieldNotFoundError", "AxisError")

import numpy


class FieldNotFoundError(IndexError):
    ...


AxisError = numpy.AxisError
