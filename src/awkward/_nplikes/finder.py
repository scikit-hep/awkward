from __future__ import annotations

from collections.abc import Collection

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


N = TypeVar("N", bound=type[NumpyLike])


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


def common_nplike(nplikes: Collection[NumpyLike]) -> NumpyLike:
    """
    Args:
        nplikes: collection of nplikes from which to determine a common nplike

    Return the common nplike for the give nplikes if such a result can be determined.
    Otherwise, raise a ValueError.
    """
    # Either we have one nplike, or one + typetracer
    if len(nplikes) == 1:
        return next(iter(nplikes))
    else:
        # We allow typetracers to mix with other nplikes, and take precedence
        for nplike in nplikes:
            if not nplike.known_data:
                return nplike

        raise ValueError(
            "cannot operate on arrays with incompatible array libraries. Use #ak.to_backend to coerce the arrays "
            "to the same backend"
        )
