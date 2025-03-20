# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.numpy_like import NumpyLike
from awkward._nplikes.virtual import VirtualArray
from awkward._typing import Any, TypeVar, cast
from awkward._util import UNSET, Sentinel

D = TypeVar("D")


_type_to_nplike: dict[type, NumpyLike] = {}
_nplike_classes: list[type[NumpyLike]] = []


N = TypeVar("N", bound="type[NumpyLike]")


def register_nplike(cls: N) -> N:
    _nplike_classes.append(cls)
    return cls


ArrayLikeT = TypeVar("ArrayLikeT", bound=ArrayLike)


def nplike_of_obj(
    obj: Any, *, default: D | Sentinel = UNSET
) -> NumpyLike[ArrayLikeT] | D:
    """
    Args:
        *arrays: iterable of possible array objects
        default: default NumpyLike instance if no array objects found

    Return the nplike that is best-suited to operating upon the given
    iterable of arrays. If no known array types are found, return `default`
    if it is set, otherwise `Numpy.instance()`.
    """

    cls = type(obj)
    try:
        return _type_to_nplike[cls]
    except KeyError:
        # Try and find the nplike for this type
        # caching the result by type
        # For virtual arrays, this is checked in the except block
        # to avoid the isinstance slowdown inside the try block
        # because it's used often for non-virtual arrays
        # TODO: replace this whole function with a more generic lookup registration system
        if isinstance(obj, VirtualArray):
            return obj.nplike
        for nplike_cls in _nplike_classes:
            if nplike_cls.is_own_array_type(cls):
                nplike = nplike_cls.instance()
                break
        else:
            if default is UNSET:
                raise TypeError(f"cannot find nplike for {cls.__name__}")
            else:
                return cast(D, default)
        _type_to_nplike[cls] = nplike
        return nplike
