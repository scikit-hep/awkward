# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def strings_astype(array, to, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        to (dtype or dtype specifier): Type to convert the strings into.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts all strings in the array to a new type, leaving the structure
    untouched.

    For example,

        >>> array = ak.Array(["1", "2", "    3    ", "00004", "-5"])
        >>> ak.strings_astype(array, np.int32)
        <Array [1, 2, 3, 4, -5] type='5 * int32'>

    and

        >>> array = ak.Array(["1.1", "2.2", "    3.3    ", "00004.4", "-5.5"])
        >>> ak.strings_astype(array, np.float64)
        <Array [1.1, 2.2, 3.3, 4.4, -5.5] type='5 * float64'>

    and finally,

        >>> array = ak.Array([["1.1", "2.2", "    3.3    "], [], ["00004.4", "-5.5"]])
        >>> ak.strings_astype(array, np.float64)
        <Array [[1.1, 2.2, 3.3], [], [4.4, -5.5]] type='3 * var * float64'>

    See also #ak.numbers_astype.
    """
    with ak._errors.OperationErrorContext(
        "ak.strings_astype",
        dict(array=array, to=to, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, to, highlevel, behavior)


def _impl(array, to, highlevel, behavior):
    to_dtype = np.dtype(to)

    def action(layout, **kwargs):
        if layout.is_list and (
            layout.parameter("__array__") == "string"
            or layout.parameter("__array__") == "bytestring"
        ):
            layout = ak.operations.without_parameters(
                layout, highlevel=False, behavior=behavior
            )
            max_length = ak.operations.max(ak.operations.num(layout, behavior=behavior))
            regulararray = ak._do.pad_none(layout, max_length, 1)
            maskedarray = ak.operations.to_numpy(regulararray, allow_missing=True)
            npstrings = maskedarray.data
            if maskedarray.mask is not False:
                npstrings[maskedarray.mask] = 0
            npnumbers = (
                npstrings.reshape(-1).view("<S" + str(max_length)).astype(to_dtype)
            )
            return ak.contents.NumpyArray(npnumbers)
        else:
            return None

    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)
    behavior = ak._util.behavior_of(array, behavior=behavior)
    out = ak._do.recursively_apply(layout, action, behavior)
    return ak._util.wrap(out, behavior, highlevel)
