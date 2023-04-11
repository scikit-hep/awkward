from __future__ import annotations

from collections.abc import Collection

from awkward._typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from awkward._nplikes.numpylike import NumpyLike

# Temporary sentinel marking "argument not given"
_UNSET = object()

D = TypeVar("D")


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


def nplike_of(*arrays, default: D = _UNSET) -> NumpyLike | D:
    """
    Args:
        *arrays: iterable of possible array objects
        default: default NumpyLike instance if no array objects found

    Return the nplike that is best-suited to operating upon the given
    iterable of arrays. If no known array types are found, return `default`
    if it is set, otherwise `Numpy.instance()`.
    """
    from awkward._nplikes.cupy import Cupy
    from awkward._nplikes.jax import Jax
    from awkward._nplikes.numpy import Numpy
    from awkward._nplikes.typetracer import TypeTracer

    nplikes: set[NumpyLike] = set()
    for array in arrays:
        if hasattr(array, "layout"):
            array = array.layout

        # Layout objects
        if hasattr(array, "backend"):
            nplikes.add(array.backend.nplike)

        # Index objects
        elif hasattr(array, "nplike"):
            nplikes.add(array.nplike)

        # Other e.g. nplike arrays
        else:
            for cls in (Numpy, Cupy, Jax, TypeTracer):
                if cls.is_own_array(array):
                    nplikes.add(cls.instance())
                    break

    if nplikes == set():
        if default is _UNSET:
            return Numpy.instance()
        else:
            return default
    else:
        return common_nplike(nplikes)
