# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._backends.backend import Backend
from awkward.contents.content import Content
from awkward.record import Record


# Note: remove_structure is semi-public, exposed via awkward.contents.content.
def remove_structure(
    layout: Content | Record,
    backend: Backend | None = None,
    flatten_records: bool = True,
    function_name: str | None = None,
    drop_nones: bool = True,
    keepdims: bool = False,
    allow_records: bool = False,
    list_to_regular: bool = False,
):
    if isinstance(layout, Record):
        return remove_structure(
            layout._array[layout._at : layout._at + 1],
            backend,
            flatten_records,
            function_name,
            drop_nones,
            keepdims,
            allow_records,
        )

    else:
        if backend is None:
            backend = layout._backend
        arrays = layout._remove_structure(
            backend,
            {
                "flatten_records": flatten_records,
                "function_name": function_name,
                "drop_nones": drop_nones,
                "keepdims": keepdims,
                "allow_records": allow_records,
                "list_to_regular": list_to_regular,
            },
        )
        return tuple(arrays)
