# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""
Compatibility module for existing users of `_typetracer
"""

from __future__ import annotations

from awkward._nplikes.typetracer import MaybeNone, OneOf  # noqa: F401

__all__ = ["UnknownScalar", "UnknownLength"]


UnknownLength = None


class _UnknownScalarMeta(type):
    def __instancecheck__(cls, instance):
        from awkward._nplikes.typetracer import is_unknown_scalar

        return is_unknown_scalar(instance)


class UnknownScalar(metaclass=_UnknownScalarMeta):
    @staticmethod
    def __new__(cls, dtype):
        from awkward._nplikes.typetracer import TypeTracer

        return TypeTracer.instance().empty((), dtype=dtype)
