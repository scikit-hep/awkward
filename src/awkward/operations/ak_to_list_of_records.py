# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_lists_of_records",)


@high_level_function()
def to_lists_of_records(array, axis=None, depth_limit=None):
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, depth_limit)


def _impl(array, axis, depth_limit):
    record_found = False

    def transform(layout, depth, **kwargs):
        nonlocal record_found
        if not layout.is_record:
            return
        if axis is None or depth == axis + 1:
            record_found = True
            obj = ak.unzip(layout, highlevel=False)
            if not layout.is_tuple:
                obj = dict(zip(layout.fields, obj))
            return ak.zip(obj, depth_limit=depth_limit, highlevel=False)

    result = ak.transform(transform, array)
    if not record_found:
        raise ValueError(f"No record found using axis={axis} in the given array")
    return result
