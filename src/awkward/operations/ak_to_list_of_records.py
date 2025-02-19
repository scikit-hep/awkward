# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_lists_of_records",)


@high_level_function()
def to_lists_of_records(array, axis=0, depth_limit=None):
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, depth_limit)


def _impl(array, axis, depth_limit):
    def transform(layout, depth, **kwargs):
        if depth == axis + 1:
            if not layout.is_record:
                raise ValueError(f"No record at axis={axis}")
            return ak.zip(
                dict(zip(layout.fields, ak.unzip(layout, highlevel=False))),
                depth_limit=depth_limit,
                highlevel=False,
            )

    return ak.transform(transform, array)
