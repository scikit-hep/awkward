from __future__ import annotations

from awkward._nplikes.numpylike import NumpyLike
from awkward._typing import Callable, TypeAlias, TypeVar

# Temporary sentinel marking "argument not given"
_UNSET = object()

D = TypeVar("D")


NumpyLikeFinder: TypeAlias = """
Callable[[type], NumpyLike | None]
"""

_type_to_nplike: dict[type, NumpyLike] = {}
_nplike_finders: list[NumpyLikeFinder] = []


N = TypeVar("N", bound="type[NumpyLike]")


def register_nplike(cls: N) -> N:
    def finder(obj_cls):
        if cls.is_own_array_type(obj_cls):
            return cls.instance()

    _nplike_finders.append(finder)
    return cls


def nplike_of(obj, *, default: D = _UNSET) -> NumpyLike | D:
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
        for finder in _nplike_finders:
            nplike = finder(cls)
            if nplike is not None:
                break
        else:
            if default is _UNSET:
                raise TypeError(f"cannot find nplike for {cls.__name__}")
            else:
                return default
        _type_to_nplike[cls] = nplike
        return nplike
