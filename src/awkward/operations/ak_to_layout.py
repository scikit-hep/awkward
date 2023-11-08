# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

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
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.typetracer import TypeTracer

__all__ = ("to_layout",)

np = NumpyMetadata.instance()
numpy = Numpy.instance()
numpy_backend = NumpyBackend.instance()
typetracer_backend = TypeTracerBackend.instance()


@high_level_function()
def to_layout(
    array,
    *,
    allow_record=True,
    allow_unknown=False,
    none_policy="error",
    use_from_iter=True,
    primitive_policy="promote",
    string_policy="as-characters",
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
        allow_unknown (bool): If True, allow non-Awkward outputs; otherwise,
            raise an error.
        use_from_iter (bool): If True, allow conversion of iterable inputs to
            arrays using #ak.from_iter; otherwise, throw an Exception.
        none_policy (bool): ("error", "pass-through", "promote"): If "error", throw an Exception
            for None inputs; if "pass-through", return None; otherwise,
            for "promote" return a length-one array containing the None with unknown type.
        primitive_policy ("error", "pass-through", "promote"): If "error", throw an Exception
            for scalar inputs; if "pass-through", return the scalar input; otherwise,
            for "promote" return a length-one array containing the scalar.
        string_policy ("error", "pass-through", "as-characters", "promote"): If "error", throw an Exception
            for scalar inputs; if "pass-through", return the scalar input;
            ir "as-characters", return an array of characters; otherwise,
            for "promote" return a length-one array containing the string.
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
        allow_unknown,
        none_policy,
        regulararray,
        use_from_iter,
        primitive_policy,
        string_policy,
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


def _handle_as_primitive(obj, layout, *, primitive_policy):
    assert primitive_policy in ("pass-through", "promote", "error")

    if primitive_policy == "pass-through":
        return obj
    elif primitive_policy == "promote":
        return layout
    elif primitive_policy == "error":
        raise TypeError(
            f"Encountered a scalar ({type(obj).__name__}), but scalar conversion/promotion is disabled"
        )
    else:
        raise ValueError(
            f"Encountered an invalid primitive policy value {primitive_policy!r}. "
            f'The permitted values are "pass-through", "promote", and "error".'
        )


def _handle_as_none(obj, layout, *, none_policy):
    assert none_policy in ("pass-through", "promote", "error")

    if none_policy == "pass-through":
        return obj
    elif none_policy == "promote":
        return layout
    elif none_policy == "error":
        raise TypeError(
            "Encountered a None value, but None conversion/promotion is disabled"
        )
    else:
        raise ValueError(
            f"Encountered an invalid none-policy value {none_policy!r}. "
            f'The permitted values are "pass-through", "promote", and "error".'
        )


def _handle_array_like(obj, layout, *, primitive_policy):
    assert primitive_policy in ("pass-through", "promote", "error")
    if obj.ndim == 0:
        return _handle_as_primitive(obj, layout, primitive_policy=primitive_policy)
    else:
        return layout


def _impl(
    obj,
    allow_record,
    allow_unknown,
    none_policy,
    regulararray,
    use_from_iter,
    primitive_policy,
    string_policy,
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
        return _handle_array_like(
            obj, promoted_layout, primitive_policy=primitive_policy
        )
    elif Cupy.is_own_array(obj):
        promoted_layout = ak.operations.from_cupy(
            obj, regulararray=regulararray, highlevel=False
        )
        return _handle_array_like(
            obj, promoted_layout, primitive_policy=primitive_policy
        )
    elif Jax.is_own_array(obj):
        promoted_layout = ak.operations.from_jax(
            obj, regulararray=regulararray, highlevel=False
        )
        return _handle_array_like(
            obj, promoted_layout, primitive_policy=primitive_policy
        )
    elif TypeTracer.is_own_array(obj):
        backend = TypeTracerBackend.instance()

        if obj.ndim == 0:
            obj = backend.nplike.reshape(obj, (1,))

        if obj.dtype.kind in {"S", "U"}:
            raise NotImplementedError(
                "strings are currently not supported for typetracer arrays"
            )

        promoted_layout = ak.contents.NumpyArray(obj, parameters=None, backend=backend)
        return _handle_array_like(
            obj, promoted_layout, primitive_policy=primitive_policy
        )
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
        return _handle_array_like(
            obj, promoted_layout, primitive_policy=primitive_policy
        )
    # Scalars
    elif isinstance(obj, (str, bytes)):
        layout = ak.operations.from_iter([obj], highlevel=False)
        if string_policy == "pass-through":
            return obj
        elif string_policy == "as-characters":
            return layout[0]
        elif string_policy == "promote":
            return layout
        elif string_policy == "error":
            raise TypeError(
                f"Encountered a {type(obj).__name__}, but string conversion/promotion is disabled"
            )
        else:
            raise ValueError(
                f"Encountered an invalid string policy value {primitive_policy!r}. "
                f'The permitted values are "pass-through", "as-characters", "promote", and "error".'
            )
    elif isinstance(obj, (datetime, date, time, Number, bool)):
        return _handle_as_primitive(
            obj,
            ak.operations.from_iter([obj], highlevel=False),
            primitive_policy=primitive_policy,
        )
    elif obj is None:
        return _handle_as_none(
            obj,
            ak.operations.from_iter([obj], highlevel=False),
            none_policy=none_policy,
        )
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
    elif allow_unknown:
        return obj
    else:
        raise TypeError(
            f"Encountered unknown type {type(obj).__name__}, and `allow_unknown` is `False`"
        )
