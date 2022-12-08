# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward.operations.ak_zeros_like import _ZEROS

np = ak._nplikes.NumpyMetadata.instance()


@ak._connect.numpy.implements("full_like")
def full_like(array, fill_value, *, dtype=None, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        fill_value: Value to fill the new array with.
        dtype (None or NumPy dtype): Overrides the data type of the result.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
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
    with ak._errors.OperationErrorContext(
        "ak.full_like",
        dict(
            array=array,
            fill_value=fill_value,
            dtype=dtype,
            highlevel=highlevel,
            behavior=behavior,
        ),
    ):
        return _impl(array, fill_value, highlevel, behavior, dtype)


def _impl(array, fill_value, highlevel, behavior, dtype):
    if dtype is not None:
        # In the case of strings and byte strings,
        # converting the fill avoids a ValueError.
        dtype = np.dtype(dtype)
        nplike = ak._nplikes.nplike_of(array)
        fill_value = nplike.array([fill_value], dtype=dtype)[0]
        # Also, if the fill_value cannot be converted to the dtype
        # this should throw a clear, early, error.
        if dtype == np.dtype(np.bool_):
            # then for bools, only 0 and 1 give correct string behavior
            fill_value = fill_value.view(np.uint8)

    layout = ak.operations.to_layout(array, allow_record=True, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)

    def action(layout, **kwargs):
        nplike = layout.backend.nplike
        index_nplike = layout.backend.index_nplike

        if layout.parameter("__array__") == "bytestring" and fill_value is _ZEROS:
            asbytes = nplike.frombuffer(b"", dtype=np.uint8)
            return ak.contents.ListArray(
                ak.index.Index64(
                    index_nplike.zeros(len(layout), dtype=np.int64), nplike=index_nplike
                ),
                ak.index.Index64(
                    index_nplike.zeros(len(layout), dtype=np.int64), nplike=index_nplike
                ),
                ak.contents.NumpyArray(asbytes, parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
            )

        elif layout.parameter("__array__") == "bytestring":
            if isinstance(fill_value, bytes):
                asbytes = fill_value
            else:
                asbytes = str(fill_value).encode("utf-8", "surrogateescape")
            asbytes = nplike.frombuffer(asbytes, dtype=np.uint8)

            return ak.contents.ListArray(
                ak.index.Index64(
                    index_nplike.zeros(len(layout), dtype=np.int64), nplike=index_nplike
                ),
                ak.index.Index64(
                    index_nplike.full(len(layout), len(asbytes), dtype=np.int64)
                ),
                ak.contents.NumpyArray(asbytes, parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
            )

        elif layout.parameter("__array__") == "string" and fill_value is _ZEROS:
            asbytes = nplike.frombuffer(b"", dtype=np.uint8)
            return ak.contents.ListArray(
                ak.index.Index64(
                    index_nplike.zeros(len(layout), dtype=np.int64), nplike=index_nplike
                ),
                ak.index.Index64(
                    index_nplike.zeros(len(layout), dtype=np.int64), nplike=index_nplike
                ),
                ak.contents.NumpyArray(asbytes, parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )

        elif layout.parameter("__array__") == "string":
            asstr = str(fill_value).encode("utf-8", "surrogateescape")
            asbytes = nplike.frombuffer(asstr, dtype=np.uint8)
            return ak.contents.ListArray(
                ak.index.Index64(
                    index_nplike.zeros(len(layout), dtype=np.int64), nplike=index_nplike
                ),
                ak.index.Index64(
                    index_nplike.full(len(layout), len(asbytes), dtype=np.int64)
                ),
                ak.contents.NumpyArray(asbytes, parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )

        elif isinstance(layout, ak.contents.NumpyArray):
            original = nplike.asarray(layout.data)

            if fill_value == 0 or fill_value is _ZEROS:
                return ak.contents.NumpyArray(
                    nplike.zeros_like(original), parameters=layout.parameters
                )
            elif fill_value == 1:
                return ak.contents.NumpyArray(
                    nplike.ones_like(original), parameters=layout.parameters
                )
            else:
                return ak.contents.NumpyArray(
                    nplike.full_like(original, fill_value), parameters=layout.parameters
                )
        else:
            return None

    out = ak._do.recursively_apply(layout, action, behavior)
    if dtype is not None:
        out = ak.operations.strings_astype(
            out, dtype, highlevel=highlevel, behavior=behavior
        )
        out = ak.operations.values_astype(
            out, dtype, highlevel=highlevel, behavior=behavior
        )
        return out
    else:
        return ak._util.wrap(out, behavior, highlevel)
