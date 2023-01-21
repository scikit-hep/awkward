from __future__ import annotations

import awkward as ak
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import ArrayLike, NumpyLike
from awkward.typing import TypeVar

# Temporary sentinel marking "argument not given"
_UNSET = object()

D = TypeVar("D")


_NPLIKE_CLASSES = (Numpy, Cupy, Jax)


def nplike_of(*arrays, default: D = _UNSET) -> NumpyLike | D:
    """
    Args:
        *arrays: iterable of possible array objects
        default: default NumpyLike instance if no array objects found

    Return the #ak._nplikes.NumpyLike that is best-suited to operating upon the given
    iterable of arrays. Return an instance of the `default_cls` if no known array types
    are found.
    """
    from awkward._typetracer import TypeTracer

    nplike_classes = _NPLIKE_CLASSES + (TypeTracer,)

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
            for cls in nplike_classes:
                if cls.is_own_array(array):
                    nplikes.add(cls.instance())
                    break

    if nplikes == set():
        if default is _UNSET:
            return Numpy.instance()
        else:
            return default
    elif len(nplikes) == 1:
        return next(iter(nplikes))
    else:
        # We allow typetracers to mix with other nplikes, and take precedence
        for nplike in nplikes:
            if not (nplike.known_data and nplike.known_shape):
                return nplike

        raise ak._errors.wrap_error(
            ValueError(
                """attempting to use arrays with more than one backend in the same operation; use
#ak.to_backend to coerce the arrays to the same backend."""
            )
        )


def to_nplike(
    array: ArrayLike, nplike: NumpyLike, *, from_nplike: NumpyLike = None
) -> ArrayLike:
    from awkward._typetracer import TypeTracer

    if from_nplike is None:
        from_nplike = nplike_of(array, default=None)
        if from_nplike is None:
            raise ak._errors.wrap_error(
                TypeError(
                    f"internal error: expected an array supported by an existing nplike, got {type(array).__name__!r}"
                )
            )

    if isinstance(from_nplike, TypeTracer) and nplike is not from_nplike:
        raise ak._errors.wrap_error(
            TypeError(
                "Converting a TypeTracer nplike to an nplike with `known_data=True` is not possible"
            )
        )

    # Copy to host memory
    if isinstance(from_nplike, Cupy):
        array = array.get()

    nplike_classes = _NPLIKE_CLASSES + (TypeTracer,)

    if isinstance(nplike, nplike_classes):
        return nplike.asarray(array)
    else:
        raise ak._errors.wrap_error(
            TypeError(f"internal error: invalid nplike {type(nplike).__name__!r}")
        )
