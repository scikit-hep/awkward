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
    def transform(layout, depth, **kwargs):
        if depth == axis + 1:
            if not layout.is_list:
                raise ValueError(f"No list at axis={axis}")
            return ak.contents.RecordArray(
                ak.unzip(layout, highlevel=False),
                layout.fields,
            )

    return ak.transform(transform, array)
