# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numbers
import os
from collections.abc import Iterable, Sequence, Sized

from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._typing import Any

np = NumpyMetadata.instance()


def is_file_path(x) -> bool:
    try:
        return os.path.isfile(x)
    except ValueError:
        return False


def is_sized_iterable(obj) -> bool:
    return isinstance(obj, Iterable) and isinstance(obj, Sized)


def is_integer(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def is_array_like(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "T")


def is_integer_like(x) -> bool:
    # Integral types
    if isinstance(x, numbers.Integral):
        return not isinstance(x, bool)
    # Scalar arrays
    elif is_array_like(x):
        return np.issubdtype(x.dtype, np.integer) and x.ndim == 0
    # Other things that support integers
    else:
        return hasattr(x, "__int__")


def is_non_string_like_iterable(obj) -> bool:
    return not isinstance(obj, (str, bytes)) and isinstance(obj, Iterable)


def is_non_string_like_sequence(obj) -> bool:
    return not isinstance(obj, (str, bytes)) and isinstance(obj, Sequence)


def regularize_axis(axis: Any, none_allowed: bool = True) -> int | None:
    """
    This function's main purpose is to convert [np,cp,...].array(0) to 0.
    """
    if is_integer_like(axis):
        regularized_axis = int(axis)
    else:
        regularized_axis = axis
    cond = is_integer(regularized_axis)
    msg = f"'axis' must be an integer, not {axis!r}"
    if none_allowed:
        cond = cond or regularized_axis is None
        msg = f"'axis' must be an integer or None, not {axis!r}"
    if not cond:
        raise TypeError(msg)
    return regularized_axis
