# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._connect.numpy import UNSUPPORTED
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, ensure_same_backend
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.typetracer import is_unknown_scalar
from awkward._regularize import is_integer_like
from awkward.operations.ak_zeros_like import _ZEROS

__all__ = ("full_like",)

np = NumpyMetadata.instance()


@high_level_function()
def full_like(
    array,
    fill_value,
    *,
    dtype=None,
    including_unknown=False,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        fill_value: Value to fill the new array with.
        dtype (None or NumPy dtype): Overrides the data type of the result.
        including_unknown (bool): If True, the `unknown` type is considered
            a value type and is converted to a zero-length array of the
            specified dtype; if False, `unknown` will remain `unknown`.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    This is the equivalent of NumPy's `np.full_like` for Awkward Arrays.

    Although it's possible to produce an array of `fill_value` with the
    structure of an `array` using #ak.broadcast_arrays:

        >>> array = ak.Array([[1, 2, 3], [], [4, 5]])
        >>> ak.broadcast_arrays(array, 1)
        [<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>,
         <Array [[1, 1, 1], [], [1, 1]] type='3 * var * int64'>]
        >>> ak.broadcast_arrays(array, 1.0)
        [<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>,
         <Array [[1, 1, 1], [], [1, 1]] type='3 * var * float64'>]

    Such a technique takes its type from the scalar (`1` or `1.0`), rather than
    the array. This function gets all types from the array, which might not be
    the same in all parts of the structure.

    Here is an extreme example:

        >>> array = ak.Array([
        ... [{"x": 0.0, "y": []},
        ...  {"x": 1.1, "y": [1]},
        ...  {"x": 2.2, "y": [1, 2]}],
        ... [],
        ... [{"x": 3.3, "y": [1, 2, None, 3]},
        ...  False,
        ...  False,
        ...  True,
        ...  {"x": 4.4, "y": [1, 2, None, 3, 4]}]])
        >>> ak.full_like(array, 12.3).show()
        [[{x: 12.3, y: []}, {x: 12.3, y: [12]}, {x: 12.3, y: [12, 12]}],
         [],
         [{x: 12.3, y: [12, 12, None, 12]}, True, ..., True, {x: 12.3, y: [12, ...]}]]

    The `"x"` values get filled in with `12.3` because they retain their type
    (`float64`) and the `"y"` list items get filled in with `12` because they
    retain their type (`int64`). Booleans get filled with True because `12.3`
    is not zero. Missing values remain in the same positions as in the original
    `array`. (To fill them in, use #ak.fill_none.)

    See also #ak.zeros_like and #ak.ones_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    # Dispatch
    yield array, fill_value

    # Implementation
    return _impl(
        array, fill_value, highlevel, behavior, dtype, including_unknown, attrs
    )


def _impl(array, fill_value, highlevel, behavior, dtype, including_unknown, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout, _ = ensure_same_backend(
            ctx.unwrap(array, primitive_policy="error"),
            ctx.unwrap(
                fill_value,
                primitive_policy="pass-through",
                string_policy="pass-through",
                allow_unknown=True,
            ),
        )

    if dtype is not None:
        # In the case of strings and byte strings,
        # converting the fill avoids a ValueError.
        dtype = np.dtype(dtype)
        fill_value = layout.backend.nplike.asarray([fill_value], dtype=dtype)[0]
        # Also, if the fill_value cannot be converted to the dtype
        # this should throw a clear, early, error.
        if dtype == np.dtype(np.bool_):
            # then for bools, only 0 and 1 give correct string behavior
            fill_value = fill_value.view(np.uint8)

    def action(layout, backend, **kwargs):
        nplike = backend.nplike
        index_nplike = backend.index_nplike

        if layout.is_numpy:
            original = nplike.asarray(layout.data)

            if fill_value is _ZEROS or (
                is_integer_like(fill_value)
                and not is_unknown_scalar(fill_value)
                and fill_value == 0
            ):
                return ak.contents.NumpyArray(
                    nplike.zeros_like(original, dtype=dtype),
                    parameters=layout.parameters,
                )
            elif (
                is_integer_like(fill_value)
                and not is_unknown_scalar(fill_value)
                and fill_value == 1
            ):
                return ak.contents.NumpyArray(
                    nplike.ones_like(original, dtype=dtype),
                    parameters=layout.parameters,
                )
            else:
                return ak.contents.NumpyArray(
                    nplike.full_like(original, fill_value, dtype=dtype),
                    parameters=layout.parameters,
                )

        elif layout.is_unknown:
            if dtype is not None and including_unknown:
                return layout.to_NumpyArray(dtype=dtype)
            else:
                return None

        elif layout.parameter("__array__") in {"bytestring", "string"}:
            stringlike_type = layout.parameter("__array__")
            charlike_type = "byte" if stringlike_type == "bytestring" else "char"

            if fill_value is _ZEROS:
                # special case because output lists will all have length zero,
                # rather than b"0" or "0" or something
                asbytes = nplike.frombuffer(b"", dtype=np.uint8)
                result = ak.contents.ListArray(
                    ak.index.Index64(
                        index_nplike.zeros(layout.length, dtype=np.int64),
                        nplike=index_nplike,
                    ),
                    ak.index.Index64(
                        index_nplike.zeros(layout.length, dtype=np.int64),
                        nplike=index_nplike,
                    ),
                    ak.contents.NumpyArray(
                        asbytes,
                        parameters={"__array__": charlike_type},
                    ),
                    parameters={"__array__": stringlike_type},
                )

            else:
                # NumPy 2.x converts "0" and b"0" (ASCII codec 48) to True (because it's not codec 0)
                # NumPy 1.x converts them to False because it parses bytestrings and strings
                # both versions of NumPy parse bytestrings and strings when converting to anything other than booleans
                numpy2_behavior = nplike.astype(nplike.asarray(["0"]), dtype=np.bool_)[
                    0
                ]
                if dtype == np.dtype(np.bool_) and numpy2_behavior:
                    asbytes = b"\1" if fill_value else b"\0"
                elif isinstance(fill_value, bytes):
                    asbytes = fill_value
                else:
                    asbytes = str(fill_value).encode("utf-8", "surrogateescape")

                asbytes = nplike.frombuffer(asbytes, dtype=np.uint8)
                result = ak.contents.ListArray(
                    ak.index.Index64(
                        index_nplike.zeros(layout.length, dtype=np.int64),
                        nplike=index_nplike,
                    ),
                    ak.index.Index64(
                        index_nplike.full(layout.length, len(asbytes), dtype=np.int64)
                    ),
                    ak.contents.NumpyArray(
                        asbytes, parameters={"__array__": charlike_type}
                    ),
                    parameters={"__array__": stringlike_type},
                )

            if dtype is not None:
                # Interpret strings as numeric/bool types
                result = ak.operations.strings_astype(
                    result, dtype, highlevel=False, behavior=behavior
                )
                # Convert dtype
                result = ak.operations.values_astype(
                    result, dtype, highlevel=False, behavior=behavior
                )
            return result
        else:
            return None

    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)


@ak._connect.numpy.implements("full_like")
def _nep_18_impl(
    a, fill_value, dtype=None, order=UNSUPPORTED, subok=UNSUPPORTED, shape=UNSUPPORTED
):
    return full_like(a, fill_value=fill_value, dtype=dtype)
