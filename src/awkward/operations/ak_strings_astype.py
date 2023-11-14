# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("strings_astype",)

np = NumpyMetadata.instance()
numpy = Numpy.instance()


@high_level_function()
def strings_astype(array, to, *, highlevel=True, behavior=None, attrs=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        to (dtype or dtype specifier): Type to convert the strings into.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
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
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, to, highlevel, behavior, attrs)


def _impl(array, to, highlevel, behavior, attrs):
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
            npnumbers = numpy.astype(
                numpy.reshape(npstrings, (-1,)).view("<S" + str(max_length)),
                dtype=np.dtype(to),
            )
            return ak.contents.NumpyArray(npnumbers)
        else:
            return None

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)
