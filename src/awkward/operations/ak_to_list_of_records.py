# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_lists_of_records",)


@high_level_function()
def to_lists_of_records(array, depth_limit=None):
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, depth_limit)


def _impl(array, depth_limit):
    def transform(layout, depth, **kwargs):
        if layout.is_record:
            obj = ak.unzip(layout, highlevel=False)
            if not layout.is_tuple:
                obj = dict(zip(layout.fields, obj))
            return ak.zip(obj, depth_limit=depth_limit, highlevel=False)

    return ak.transform(transform, array)
