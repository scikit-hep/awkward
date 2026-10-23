# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""NumPy overloads for flat numeric Awkward views inside Numba."""

import numba
import numpy

from awkward._connect.numba.arrayview import ArrayViewType, array_supported
from awkward._connect.numba.layout import NumpyArrayType


def _is_flat_numeric(value):
    return (
        isinstance(value, ArrayViewType)
        and isinstance(value.type, NumpyArrayType)
        and value.type.ndim == 1
        and array_supported(value.type.inner_dtype)
    )


def _register_unary(function):
    @numba.extending.overload(function)
    def overload(a):
        if _is_flat_numeric(a):

            def impl(a):
                return function(numpy.asarray(a))

            return impl


def _register_axis_dtype(function):
    @numba.extending.overload(function)
    def overload(a, axis=None, dtype=None):
        if _is_flat_numeric(a):

            def impl(a, axis=None, dtype=None):
                return function(numpy.asarray(a), axis=axis, dtype=dtype)

            return impl


_register_axis_dtype(numpy.sum)

for _function in (
    numpy.prod,
    numpy.min,
    numpy.max,
    numpy.mean,
    numpy.std,
    numpy.var,
    numpy.median,
    numpy.argmin,
    numpy.argmax,
    numpy.all,
    numpy.any,
    numpy.cumsum,
    numpy.cumprod,
    numpy.sort,
    numpy.argsort,
):
    _register_unary(_function)
