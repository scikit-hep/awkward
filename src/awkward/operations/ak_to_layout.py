# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("to_layout",)

from collections.abc import Iterable
from datetime import date, datetime, time
from numbers import Number
from typing import Any

from awkward_cpp.lib import _ext

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend
from awkward._dispatch import high_level_function
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.typetracer import TypeTracer

np = NumpyMetadata.instance()
numpy = Numpy.instance()
numpy_backend = NumpyBackend.instance()
typetracer_backend = TypeTracerBackend.instance()


@high_level_function()
def to_layout(
    array,
    *,
    allow_record=True,
    allow_other=False,
    allow_none=False,
    use_from_iter=True,
    scalar_policy="promote",
    string_as_characters=True,
    regulararray=True,
):
    """
    Args:
        array: Array-like data. May be a high level #ak.Array, #ak.Record (if `allow_record`),
            #ak.ArrayBuilder, or low-level #ak.contents.Content, #ak.record.Record (if `allow_record`),
            or a supported backend array (NumPy `ndarray`, CuPy `ndarray`,
            JAX DeviceArray), data-less TypeTracer, Arrow object, or an arbitrary Python
            iterable (for #ak.from_iter to convert).
        allow_record (bool): If True, allow #ak.record.Record as an output;
            otherwise, if the output would be a scalar record, raise an error.
        allow_other (bool): If True, allow non-Awkward outputs; otherwise,
            raise an error.
        allow_none (bool): If True, allow None outputs; otherwise, raise an
            error.
        use_from_iter (bool): If True, allow conversion of iterable inputs to
            arrays using #ak.from_iter; otherwise, throw an Exception.
        scalar_policy ("error", "allow", "promote"): If "error", throw an Exception
            for scalar inputs; if "allow", return the scalar input; otherwise,
            for "promote" return a length-one array containing the scalar.
        string_as_characters (bool): If True, scalar strings are converted
            to an array of characters; otherwise, scalar strings are taken to be
            the Python object itself.
        regulararray (bool): Prefer to create #ak.contents.RegularArray nodes for
            regular array objects.

    Converts `array` (many types supported, including all Awkward Arrays and
    Records) into a #ak.contents.Content and maybe #ak.record.Record or
    other types.

    This function is usually used to sanitize inputs for other functions; it
    would rarely be used in a data analysis because #ak.contents.Content and
    #ak.record.Record are lower-level than #ak.Array.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(
        array,
        allow_record,
        allow_other,
        allow_none,
        regulararray,
        use_from_iter,
        scalar_policy,
        string_as_characters,
    )


def maybe_merge_mappings(primary, secondary):
    if secondary is None:
        return primary
    elif primary is None:
        return secondary
    elif primary is secondary:
        return primary
    else:
        return {**primary, **secondary}


def _handle_as_scalar(obj, *, scalar_policy):
    assert scalar_policy in ("allow", "promote", "error")

    if scalar_policy == "allow":
        return obj
    elif scalar_policy == "promote":
        layout = ak.operations.from_iter([obj], highlevel=False)
        return layout
    else:
        assert scalar_policy == "error"
        raise TypeError(
            f"Encountered a scalar ({type(obj).__name__}), but scalars conversion/promotion is disabled"
        )


def _handle_array_like(obj, layout, *, scalar_policy):
    assert scalar_policy in ("allow", "promote", "error")
    if obj.ndim == 0:
        if scalar_policy == "allow":
            return obj
        elif scalar_policy == "promote":
            return layout
        else:
            assert scalar_policy == "error"
            raise TypeError(
                f"Encountered a scalar ({type(obj).__name__}), but scalars conversion/promotion is disabled"
            )
    else:
        return layout


def _impl(
    obj,
    allow_record,
    allow_other,
    allow_none,
    regulararray,
    use_from_iter,
    scalar_policy,
    string_as_characters,
) -> Any:
    # Well-defined types
    if isinstance(obj, ak.contents.Content):
        return obj
    elif isinstance(obj, ak.record.Record):
        if not allow_record:
            raise TypeError("ak.Record objects are not allowed in this function")
        else:
            return obj
    elif isinstance(obj, ak.highlevel.Array):
        return obj.layout
    elif isinstance(obj, ak.highlevel.Record):
        if not allow_record:
            raise TypeError("ak.Record objects are not allowed in this function")
        else:
            return obj.layout
    elif isinstance(obj, ak.highlevel.ArrayBuilder):
        return obj.snapshot().layout
    elif isinstance(obj, _ext.ArrayBuilder):
        return obj.snapshot()
    elif numpy.is_own_array(obj):
        promoted_layout = ak.operations.from_numpy(
            obj,
            regulararray=regulararray,
            recordarray=True,
            highlevel=False,
        )
        return _handle_array_like(obj, promoted_layout, scalar_policy=scalar_policy)
    elif Cupy.is_own_array(obj):
        promoted_layout = ak.operations.from_cupy(
            obj, regulararray=regulararray, highlevel=False
        )
        return _handle_array_like(obj, promoted_layout, scalar_policy=scalar_policy)
    elif Jax.is_own_array(obj):
        promoted_layout = ak.operations.from_jax(
            obj, regulararray=regulararray, highlevel=False
        )
        return _handle_array_like(obj, promoted_layout, scalar_policy=scalar_policy)
    elif TypeTracer.is_own_array(obj):
        backend = TypeTracerBackend.instance()

        if obj.ndim == 0:
            obj = backend.nplike.reshape(obj, (1,))

        if obj.dtype.kind in {"S", "U"}:
            raise NotImplementedError(
                "strings are currently not supported for typetracer arrays"
            )

        promoted_layout = ak.contents.NumpyArray(obj, parameters=None, backend=backend)
        return _handle_array_like(obj, promoted_layout, scalar_policy=scalar_policy)
    elif ak._util.in_module(obj, "pyarrow"):
        return ak.operations.from_arrow(obj, highlevel=False)
    elif hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__"):
        return ak.operations.from_dlpack(obj, highlevel=False)
    # Typed scalars
    elif isinstance(obj, np.generic):
        promoted_layout = ak.operations.from_numpy(
            numpy.asarray(obj),
            regulararray=regulararray,
            recordarray=True,
            highlevel=False,
        )
        return _handle_array_like(obj, promoted_layout, scalar_policy=scalar_policy)
    # Scalars
    elif isinstance(obj, (str, bytes)):
        layout = ak.operations.from_iter([obj], highlevel=False)
        if scalar_policy == "allow":
            if string_as_characters:
                return layout[0]
            else:
                return obj
        elif scalar_policy == "promote":
            return layout
        else:
            raise TypeError(
                f"Encountered a scalar ({type(obj).__name__}), but scalars conversion/promotion is disabled"
            )
    elif isinstance(obj, (datetime, date, time, Number, bool)):
        return _handle_as_scalar(obj, scalar_policy=scalar_policy)
    elif obj is None:
        if allow_none:
            return _handle_as_scalar(obj, scalar_policy=scalar_policy)
        else:
            raise TypeError("Encountered None value, and `allow_none` is `False`")
    # Iterables
    elif isinstance(obj, Iterable):
        if use_from_iter:
            return ak.operations.from_iter(
                obj, highlevel=False, allow_record=allow_record
            )
        else:
            raise TypeError(
                "Encountered an iterable object, but coercing iterables is disabled"
            )
    # Unknown types
    elif allow_other:
        return obj
    else:
        raise TypeError(
            f"Encountered unknown type {type(obj).__name__}, and `allow_other` is `False`"
        )
