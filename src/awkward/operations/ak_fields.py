# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("fields",)

np = NumpyMetadata.instance()


@high_level_function()
def fields(array):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Extracts record fields or tuple slot numbers from `array` (many types
    supported, including all Awkward Arrays and Records).

    If the array contains nested records, only the outermost record is
    queried. If it contains tuples instead of records, this function outputs
    string representations of integers, such as `"0"`, `"1"`, `"2"`, etc.
    The records or tuples may be within multiple layers of nested lists.

    If the array contains neither tuples nor records, this returns an empty
    list.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    layout = ak.operations.to_layout(array, allow_record=True, primitive_policy="error")
    return layout.fields.copy()
