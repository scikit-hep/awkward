# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("values_astype",)

np = NumpyMetadata.instance()


@high_level_function()
def values_astype(
    array, to, *, including_unknown=False, highlevel=True, behavior=None, attrs=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        to (dtype or dtype specifier): Type to convert the numbers into.
        including_unknown (bool): If True, the `unknown` type is considered
            a value type and is converted to the specified dtype; if False,
            `unknown` will remain `unknown`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Converts all numbers in the array to a new type, leaving the structure
    untouched.

    For example,

        >>> array = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
        >>> ak.values_astype(array, np.int32)
        <Array [1, 2, 3, 4, 5] type='5 * int32'>

    and

        >>> array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        >>> ak.values_astype(array, np.int32)
        <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int32'>

    Note, when converting values to a `np.datetime64` type that is unitless, a
    default '[us]' unit is assumed - until further specified as numpy dtypes.

    For example,

        >>> array = ak.Array([1567416600000])
        >>> ak.values_astype(array, "datetime64[ms]")
        <Array [2019-09-02T09:30:00.000] type='1 * datetime64[ms]'>

    or

        >>> array = ak.Array([1567416600000])
        >>> ak.values_astype(array, np.dtype("M8[ms]"))
        <Array [2019-09-02T09:30:00.000] type='1 * datetime64[ms]['>

    See also #ak.strings_astype.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, to, including_unknown, highlevel, behavior, attrs)


def _impl(array, to, including_unknown, highlevel, behavior, attrs):
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")
    to_str = ak.types.numpytype.dtype_to_primitive(np.dtype(to))
    out = ak._do.numbers_to_type(layout, to_str, including_unknown)
    return ctx.wrap(out, highlevel=highlevel)
