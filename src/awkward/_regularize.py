# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import numbers
import os
from collections.abc import Iterable, Sequence, Sized

from awkward._nplikes.numpylike import NumpyMetadata
from awkward._typing import AxisMaybeNone, SupportsInt

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


def regularize_path(path):
    """
    Converts pathlib Paths into plain string paths (for all versions of Python).
    """
    is_path = False

    if isinstance(path, getattr(os, "PathLike", ())):
        is_path = True
        path = os.fspath(path)

    elif hasattr(path, "__fspath__"):
        is_path = True
        path = path.__fspath__()

    elif path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(path, pathlib.Path):
            is_path = True
            path = str(path)

    return is_path, path


def regularize_axis(axis: SupportsInt | None) -> AxisMaybeNone:
    if axis is None:
        return None
    else:
        return int(axis)
