from __future__ import annotations

from awkward.typing import Callable, TypeAlias, TypeVar

from .numpylike import NumpyLike

T = TypeVar("T")
NumpyLikeFinder: TypeAlias = "Callable[[T], NumpyLike]"
NumpyLikeFinderFactory: TypeAlias = "Callable[[type[T]], NumpyLikeFinder[T] | None]"

_type_finders: dict[type, NumpyLikeFinder] = {}
_finder_factories = []


def register_nplike_finder_factory(finder: NumpyLikeFinderFactory):
    """
    Args:
        finder: nplike finder callable

    Returns the given finder, after registering it in a list of nplike finders.

    """
    _finder_factories.append(finder)
    return finder


N = TypeVar("N", bound=type[NumpyLike])


def register_nplike(cls: N) -> N:
    """
    Args:
        cls: nplike class

    Returns the given class, after registering `cls.is_own_array` as an nplike
    finder.
    """

    def nplike_finder_factory(type_):
        if cls.is_own_array_type(type_):

            def primitive_finder(_):
                return cls.instance()

            return primitive_finder

    register_nplike_finder_factory(nplike_finder_factory)
    return cls


def find_nplike_for(obj) -> NumpyLike | None:
    """
    Args:
        obj: object for which to find an nplike

    Return the first nplike for which a finder returns a non-None result.
    """
    cls = type(obj)
    try:
        finder = _type_finders[cls]
    except KeyError:
        for factory in _finder_factories:
            maybe_finder = factory(cls)
            if maybe_finder is not None:
                break
        else:
            return None
        _type_finders[cls] = maybe_finder
        return maybe_finder(obj)
    else:
        return finder(obj)
