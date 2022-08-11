# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from awkward._v2.operations.ak_zeros_like import _ZEROS
import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


# @ak._v2._connect.numpy.implements("full_like")
def full_like(array, fill_value, highlevel=True, behavior=None, dtype=None):
    """
    Args:
        array: Array to use as a model for a replacement that contains only
            `fill_value`.
        fill_value: Value to fill new new array with.
        highlevel (bool, default is True): If True, return an #ak.Array;
            otherwise, return a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        dtype (None or NumPy dtype)): Overrides the data type of the result.

    This is the equivalent of NumPy's `np.full_like` for Awkward Arrays.

    Although it's possible to produce an array of `fill_value` with the structure
    of an `array` using #ak.broadcast_arrays:

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
        >>> ak.to_list(ak.full_like(array, 12.3))
        [[{"x": 12.3, "y": []},
          {"x": 12.3, "y": [12]},
          {"x": 12.3, "y": [12, 12]}],
         [],
         [{"x": 12.3, "y": [12, 12, None, 12]},
          True,
          True,
          True,
          {"x": 12.3, "y": [12, 12, None, 12, 12]}]]

    The `"x"` values get filled in with `12.3` because they retain their type
    (`float64`) and the `"y"` list items get filled in with `12` because they
    retain their type (`int64`). Booleans get filled with True because `12.3`
    is not zero. Missing values remain in the same positions as in the original
    `array`. (To fill them in, use #ak.fill_none.)

    See also #ak.zeros_like and #ak.ones_like.

    (There is no equivalent of NumPy's `np.empty_like` because Awkward Arrays
    are immutable.)
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.full_like",
        dict(
            array=array,
            fill_value=fill_value,
            highlevel=highlevel,
            behavior=behavior,
            dtype=dtype,
        ),
    ):
        return _impl(array, fill_value, highlevel, behavior, dtype)


def _impl(array, fill_value, highlevel, behavior, dtype):
    if dtype is not None:
        # In the case of strings and byte strings,
        # converting the fill avoids a ValueError.
        dtype = np.dtype(dtype)
        nplike = ak.nplike.of(array)
        fill_value = nplike.array([fill_value], dtype=dtype)[0]
        # Also, if the fill_value cannot be converted to the dtype
        # this should throw a clear, early, error.
        if dtype == np.dtype(np.bool_):
            # then for bools, only 0 and 1 give correct string behavior
            fill_value = fill_value.view(np.uint8)

    layout = ak._v2.operations.to_layout(array, allow_record=True, allow_other=False)

    def action(layout, **kwargs):
        if layout.parameter("__array__") == "bytestring" and fill_value is _ZEROS:
            nplike = ak.nplike.of(layout)
            asbytes = nplike.frombuffer(b"", dtype=np.uint8)
            return ak._v2.contents.ListArray(
                ak._v2.index.Index64(
                    nplike.index_nplike.zeros(len(layout), dtype=np.int64),
                    nplike=nplike,
                ),
                ak._v2.index.Index64(
                    nplike.index_nplike.zeros(len(layout), dtype=np.int64),
                    nplike=nplike,
                ),
                ak._v2.contents.NumpyArray(asbytes, parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
            )

        elif layout.parameter("__array__") == "bytestring":
            nplike = ak.nplike.of(layout)
            if isinstance(fill_value, bytes):
                asbytes = fill_value
            else:
                asbytes = str(fill_value).encode("utf-8", "surrogateescape")
            asbytes = nplike.frombuffer(asbytes, dtype=np.uint8)

            return ak._v2.contents.ListArray(
                ak._v2.index.Index64(
                    nplike.index_nplike.zeros(len(layout), dtype=np.int64),
                    nplike=nplike,
                ),
                ak._v2.index.Index64(
                    nplike.index_nplike.full(len(layout), len(asbytes), dtype=np.int64)
                ),
                ak._v2.contents.NumpyArray(asbytes, parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
            )

        elif layout.parameter("__array__") == "string" and fill_value is _ZEROS:
            nplike = ak.nplike.of(layout)
            asbytes = nplike.frombuffer(b"", dtype=np.uint8)
            return ak._v2.contents.ListArray(
                ak._v2.index.Index64(
                    nplike.index_nplike.zeros(len(layout), dtype=np.int64),
                    nplike=nplike,
                ),
                ak._v2.index.Index64(
                    nplike.index_nplike.zeros(len(layout), dtype=np.int64),
                    nplike=nplike,
                ),
                ak._v2.contents.NumpyArray(asbytes, parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )

        elif layout.parameter("__array__") == "string":
            nplike = ak.nplike.of(layout)
            asstr = str(fill_value).encode("utf-8", "surrogateescape")
            asbytes = nplike.frombuffer(asstr, dtype=np.uint8)
            return ak._v2.contents.ListArray(
                ak._v2.index.Index64(
                    nplike.index_nplike.zeros(len(layout), dtype=np.int64),
                    nplike=nplike,
                ),
                ak._v2.index.Index64(
                    nplike.index_nplike.full(len(layout), len(asbytes), dtype=np.int64)
                ),
                ak._v2.contents.NumpyArray(asbytes, parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )

        elif isinstance(layout, ak._v2.contents.NumpyArray):
            nplike = ak.nplike.of(layout)
            original = nplike.asarray(layout.data)

            if fill_value == 0 or fill_value is _ZEROS:
                return ak._v2.contents.NumpyArray(
                    nplike.zeros_like(original),
                    layout.identifier,
                    layout.parameters,
                )
            elif fill_value == 1:
                return ak._v2.contents.NumpyArray(
                    nplike.ones_like(original),
                    layout.identifier,
                    layout.parameters,
                )
            else:
                return ak._v2.contents.NumpyArray(
                    nplike.full_like(original, fill_value),
                    layout.identifier,
                    layout.parameters,
                )
        else:
            return None

    out = layout.recursively_apply(action, behavior)
    if dtype is not None:
        out = ak._v2.operations.strings_astype(out, dtype, highlevel, behavior)
        out = ak._v2.operations.values_astype(out, dtype, highlevel, behavior)
        return out
    return ak._v2._util.wrap(out, behavior, highlevel)
