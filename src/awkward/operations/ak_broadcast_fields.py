# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()


def broadcast_fields(
    *arrays,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        arrays: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Return a list of arrays whose types contain the same number of fields. Unlike
    #ak.broadcast_arrays, this function does not require record types to occur at the
    same depths. Where fields are missing from a record ...
    """
    with ak._errors.OperationErrorContext(
        "ak.broadcast_fields",
        {
            "arrays": arrays,
            "highlevel": highlevel,
            "behavior": behavior,
        },
    ):
        return _impl(arrays, highlevel, behavior)


def _impl(arrays, highlevel, behavior):
    def descend_to_record(layout):
        if layout.is_list:
            return layout.copy(content=descend_to_record(layout.content))
        elif layout.is_record:
            return layout
        elif layout.is_option:
            return layout.copy(content=descend_to_record(layout.content))
        elif layout.is_indexed:
            return layout.copy(content=descend_to_record(layout.content))
        elif layout.is_leaf:
            return layout
        elif layout.is_union:
            raise ak._errors.wrap_error(TypeError("encountered union"))
        else:
            raise ak._errors.wrap_error(AssertionError("unexpected content type"))

    def recurse(inputs):
        records = [descend_to_record(x) for x in inputs]

        if not all(layout.is_record for layout in records):
            return records

        fields = ak._util.unique_list([f for layout in records for f in layout.fields])

        # For each field, build a union over the zero-length arrays of all layouts
        field_unions = []
        for field in fields:
            field_contents = []
            for layout in records:
                field_content = layout.maybe_content(field)
                field_contents.append(
                    field_content.form.length_zero_array(
                        backend=field_content.backend, highlevel=False
                    )
                )
            field_unions.append(ak._do.merge_as_union(field_contents))

        # For each layout, build a record built from unions. Each union is formed over
        # the full-length field layout and the zero-length field union
        next_layouts = []
        for layout in records:
            if layout.fields == fields:
                next_layouts.append(layout)
            else:
                index_nplike = layout.backend.index_nplike
                tags = ak.index.Index8(index_nplike.zeros(layout.length, dtype=np.int8))
                index = ak.index.Index64(
                    index_nplike.arange(layout.length, dtype=np.int64)
                )
                next_layout = layout.copy(
                    fields=fields,
                    contents=[
                        ak.contents.UnionArray.simplified(
                            tags=tags,
                            index=index,
                            contents=[layout.maybe_content(field), field_union],
                        )
                        for field, field_union in zip(fields, field_unions)
                    ],
                )
                next_layouts.append(next_layout)
        return next_layouts

    layouts = [ak.to_layout(x) for x in arrays]
    return [
        ak._util.wrap(x, highlevel=highlevel, behavior=behavior)
        for x in recurse(layouts)
    ]
