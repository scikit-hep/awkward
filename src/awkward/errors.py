# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy

__all__ = ("AxisError", "ExperimentalWarning", "FieldNotFoundError")


class ExperimentalWarning(UserWarning):
    """Issued on first use of an API that may change or be removed in any
    release, without a deprecation period."""


class FieldNotFoundError(IndexError):
    pass


AxisError = getattr(numpy, "exceptions", numpy).AxisError
