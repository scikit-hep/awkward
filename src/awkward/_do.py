# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

from awkward.contents.content import ActionType, Content
from awkward.record import Record
from awkward.typing import Any


def recursively_apply(
    layout: Content | Record,
    action: ActionType,
    behavior: dict | None = None,
    depth_context: dict[str, Any] | None = None,
    lateral_context: dict[str, Any] | None = None,
    allow_records: bool = True,
    keep_parameters: bool = True,
    numpy_to_regular: bool = True,
    return_simplified: bool = True,
    return_array: bool = True,
    function_name: str | None = None,
) -> Content | Record | None:

    if isinstance(layout, Content):
        return layout._recursively_apply(
            action,
            behavior,
            1,
            copy.copy(depth_context),
            lateral_context,
            {
                "allow_records": allow_records,
                "keep_parameters": keep_parameters,
                "numpy_to_regular": numpy_to_regular,
                "return_simplified": return_simplified,
                "return_array": return_array,
                "function_name": function_name,
            },
        )

    elif isinstance(layout, Record):
        out = recursively_apply(
            layout._array,
            action,
            behavior,
            depth_context,
            lateral_context,
            allow_records,
            keep_parameters,
            numpy_to_regular,
            return_simplified,
            return_array,
            function_name,
        )

        if return_array:
            return Record(out, layout.at)
        else:
            return None
