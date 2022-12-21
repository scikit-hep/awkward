import awkward as ak
from awkward._nplikes import (
    Array,
    ArrayModuleNumpyLike,
    Cupy,
    Jax,
    Numpy,
    NumpyLike,
    TypeTracer,
)
from awkward.typing import TypeVar

# Temporary sentinel marking "argument not given"
_UNSET = object()

D = TypeVar("D")


def nplike_of(*arrays, default: D = _UNSET) -> NumpyLike | D:
    """
    Args:
        *arrays: iterable of possible array objects
        default: default NumpyLike instance if no array objects found

    Return the #ak._nplikes.NumpyLike that is best-suited to operating upon the given
    iterable of arrays. Return an instance of the `default_cls` if no known array types
    are found.
    """
    nplikes: set[NumpyLike] = set()
    nplike_classes = (Numpy, Cupy, Jax, TypeTracer)
    for array in arrays:
        # Highlevel objecys
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
                # ArrayModule-like nplikes can wrap arrays
                if issubclass(cls, ArrayModuleNumpyLike) and cls.is_own_or_raw_array(
                    array
                ):
                    nplikes.add(cls.instance())
                    break
                elif cls.is_own_array(array):
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


S = TypeVar("S", bound=Array)
T = TypeVar("T", bound=Array)


def convert(from_nplike: NumpyLike[S], to_nplike: NumpyLike[T], array: S) -> T:
    if isinstance(from_nplike, TypeTracer):
        raise ak._errors.wrap_error(
            TypeError("typetracer arrays cannot be converted to other nplikes")
        )
    else:
        return to_nplike.asarray(array)
