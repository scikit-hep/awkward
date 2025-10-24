# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy

__all__ = ("AxisError", "FieldNotFoundError")


class FieldNotFoundError(IndexError):
    pass


AxisError = getattr(numpy, "exceptions", numpy).AxisError
