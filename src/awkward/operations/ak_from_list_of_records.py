# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_lists_of_records",)


@high_level_function()
def from_lists_of_records(array, axis=0):
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis)


def _impl(array, axis):
    list_found = False

    def transform(layout, depth, **kwargs):
        nonlocal list_found
        if not layout.is_list:
            return
        if axis is None or depth == axis + 1:
            list_found = True
            return ak.contents.RecordArray(
                ak.unzip(layout, highlevel=False),
                None if layout.is_tuple else layout.fields,
            )

    result = ak.transform(transform, array)
    if not list_found:
        raise ValueError(f"No list found using axis={axis} in the given array")
    return result
