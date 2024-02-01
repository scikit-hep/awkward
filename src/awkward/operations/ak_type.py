# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numbers
from datetime import datetime, timedelta

from awkward_cpp.lib import _ext

import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("type",)

np = NumpyMetadata.instance()


@high_level_function()
def type(array, *, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        behavior (None or dict): Custom #ak.behavior for the output type, if
            high-level.

    The high-level type of an `array` (many types supported, including all
    Awkward Arrays and Records) as #ak.types.Type objects.

    The high-level type ignores layout differences like
    #ak.contents.ListArray versus #ak.contents.ListOffsetArray, but
    not differences like "regular-sized lists" (i.e.
    #ak.contents.RegularArray) versus "variable-sized lists" (i.e.
    #ak.contents.ListArray and similar).

    Types are rendered as [Datashape](https://datashape.readthedocs.io/)
    strings, which makes the same distinctions.

    For example,

        >>> array = ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
        ...                   [],
        ...                   [{"x": 3.3, "y": [3, 3, 3]}]])

    has type

        >>> ak.type(array).show()
        3 * var * {
            x: float64,
            y: var * int64
        }

    but

        >>> array = ak.Array(np.arange(2*3*5).reshape(2, 3, 5))

    has type

        >>> ak.type(array).show()
        2 * 3 * 5 * int64

    Some cases, like heterogeneous data, require [extensions beyond the
    Datashape specification](https://github.com/blaze/datashape/issues/237).
    For example,

        >>> array = ak.Array([1, "two", [3, 3, 3]])

    has type

        >>> ak.type(array).show()
        3 * union[
            int64,
            string,
            var * int64
        ]

    but "union" is not a Datashape type-constructor. (Its syntax is
    similar to existing type-constructors, so it's a plausible addition
    to the language.)
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, behavior)


def _impl(array, behavior):
    behavior = behavior_of(array, behavior=behavior)

    if isinstance(array, (ak.highlevel.Record, ak.highlevel.Array)):
        return _impl(array.layout, behavior)

    elif isinstance(array, _ext.ArrayBuilder):
        form = ak.forms.from_json(array.form())
        return ak.types.ArrayType(form.type, len(array), behavior=behavior)

    elif isinstance(array, ak.record.Record):
        return ak.types.ScalarType(array.array.form.type, behavior=behavior)

    elif isinstance(array, ak.contents.Content):
        return ak.types.ArrayType(array.form.type, array.length, behavior=behavior)

    elif isinstance(array, (np.dtype, np.generic)):
        return ak.types.ScalarType(
            ak.types.NumpyType(ak.types.numpytype.dtype_to_primitive(np.dtype(array)))
        )

    elif isinstance(array, bool):  # np.bool_ in np.generic (above)
        return ak.types.ScalarType(ak.types.NumpyType("bool"), behavior=behavior)

    elif isinstance(array, (str, bytes)):
        return ak.types.ArrayType(
            ak.types.NumpyType(
                "uint8",
                parameters={"__array__": "char" if isinstance(array, str) else "byte"},
            ),
            len(array),
            behavior=behavior,
        )

    elif isinstance(array, numbers.Integral):
        return ak.types.ScalarType(ak.types.NumpyType("int64"), behavior=behavior)

    elif isinstance(array, numbers.Real):
        return ak.types.ScalarType(ak.types.NumpyType("float64"), behavior=behavior)

    elif isinstance(array, numbers.Complex):
        return ak.types.ScalarType(ak.types.NumpyType("complex128"), behavior=behavior)

    elif isinstance(array, datetime):  # np.datetime64 in np.generic (above)
        return ak.types.ScalarType(ak.types.NumpyType("datetime64"), behavior=behavior)

    elif isinstance(array, timedelta):  # np.timedelta64 in np.generic (above)
        return ak.types.ScalarType(ak.types.NumpyType("timedelta"), behavior=behavior)

    elif isinstance(array, ak.highlevel.ArrayBuilder):
        # Don't go through `to_layout`: we want to avoid snapshotting this array
        return _impl(array._layout, behavior)

    elif array is None:
        return ak.types.ScalarType(ak.types.UnknownType(), behavior=behavior)

    else:
        layout = ak.to_layout(array, allow_record=False)
        return _impl(layout, behavior)
