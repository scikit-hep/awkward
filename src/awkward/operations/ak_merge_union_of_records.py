# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak
from awkward._nplikes.numpylike import NumpyMetadata

np = NumpyMetadata.instance()
cpu = ak._backends.NumpyBackend.instance()


def merge_union_of_records(array, axis=-1, *, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied.
            The outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the  innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Simplifies unions of records, e.g.

        >>> array = ak.Array([{"a": 1}, {"b": 2}])

    into records of options, i.e.

        >>> ak.merge_union_of_records(array)
        <Array [{a: 1, b: None}, {a: None, ...}] type='2 * {a: ?int64, b: ?int64}'>
    """
    with ak._errors.OperationErrorContext(
        "ak.merge_union_of_records",
        {"array": array, "axis": axis, "highlevel": highlevel, "behavior": behavior},
    ):
        return _impl(array, axis, highlevel, behavior)


def _impl(array, axis, highlevel, behavior):
    behavior = ak._util.behavior_of(array, behavior=behavior)
    layout = ak.to_layout(array, allow_record=False)

    def apply_displace_index(layout, backend, **kwargs):
        if layout.is_record:
            return layout
        elif layout.is_option and layout.content.is_record:
            raise ak._errors.wrap_error(
                TypeError(
                    "optional records cannot be merged by this function. First call `ak.merge_option_of_records` "
                    "to convert these into records of options."
                )
            )
        elif layout.is_indexed and layout.content.is_record:
            record = layout.content
            # Transpose index-of-record to record-of-index
            return ak.contents.RecordArray(
                [
                    ak.contents.IndexedArray.simplified(
                        layout.index, c, parameters=layout._parameters
                    )
                    for c in record.contents
                ],
                record.fields,
                record.length,
                backend=backend,
            )
        else:
            raise ak._errors.wrap_error(TypeError(layout))

    def apply(layout, depth, backend, **kwargs):
        posaxis = ak._util.maybe_posaxis(layout, axis, depth)
        if depth < posaxis + 1 and layout.is_leaf:
            raise ak._errors.wrap_error(
                np.AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
            )
        elif depth == posaxis + 1 and layout.is_union:
            if all(x.is_record for x in layout.contents):
                # First, find all ordered fields, regularising any index-of-record
                # such that we have record-of-index
                seen_fields = set()
                all_fields = []
                regularised_contents = []
                for content in layout.contents:
                    # Ensure that we have record-of-index
                    regularised_content = ak._do.recursively_apply(
                        content, apply_displace_index
                    )
                    regularised_contents.append(regularised_content)

                    # Find new fields
                    for field in regularised_content.fields:
                        if field not in seen_fields:
                            seen_fields.add(field)
                            all_fields.append(field)

                # Build unions for each field
                outer_field_contents = []
                for field in all_fields:
                    field_tags = backend.index_nplike.asarray(layout.tags, copy=True)
                    field_index = backend.index_nplike.asarray(layout.index, copy=True)

                    # Build contents for union representing current field
                    field_contents = [
                        c.content(field)
                        for c in regularised_contents
                        if c.has_field(field)
                    ]

                    # Find the best location for option type.
                    # We will potentially have fewer contents in this per-field union
                    # than the original outer union-of-records, because some recordarrays
                    # may not have the given field.
                    tag_for_missing = 0
                    for i, content in enumerate(field_contents):
                        if content.is_option:
                            tag_for_missing = i
                            break

                    # If at least one recordarray doesn't have this field, we add
                    # a special option
                    if len(field_contents) < len(regularised_contents):
                        # Make the tagged content an option, growing by one to ensure we
                        # have a known `None` value to index into
                        tagged_content = field_contents[tag_for_missing]
                        indexedoption_index = backend.index_nplike.arange(
                            tagged_content.length + 1, dtype=np.int64
                        )
                        indexedoption_index[tagged_content.length] = -1
                        field_contents[
                            tag_for_missing
                        ] = ak.contents.IndexedOptionArray.simplified(
                            ak.index.Index64(indexedoption_index), tagged_content
                        )

                    # Now build contents for union, by looping over outermost index
                    # Overwrite tags to adjust for new contents length
                    # and use the tagged content for any missing values
                    k = 0
                    for j, content in enumerate(regularised_contents):
                        tag_is_j = field_tags == j

                        if content.has_field(field):
                            # Rewrite tags to account for missing fields
                            field_tags[tag_is_j] = k
                            k += 1

                        else:
                            # Rewrite tags to point to option content
                            field_tags[tag_is_j] = tag_for_missing
                            # Point each value to missing value
                            field_index[tag_is_j] = (
                                field_contents[tag_for_missing].length - 1
                            )

                    outer_field_contents.append(
                        ak.contents.UnionArray.simplified(
                            ak.index.Index8(field_tags),
                            ak.index.Index64(field_index),
                            field_contents,
                        )
                    )
                return ak.contents.RecordArray(
                    outer_field_contents, all_fields, backend=backend
                )

    out = ak._do.recursively_apply(layout, apply)
    return ak._util.wrap(out, highlevel=highlevel, behavior=behavior)
