# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext, maybe_posaxis
from awkward._nplikes.numpy_like import ArrayLike, NumpyMetadata
from awkward._regularize import regularize_axis
from awkward.errors import AxisError

__all__ = ("merge_union_of_records",)

np = NumpyMetadata.instance()


@high_level_function()
def merge_union_of_records(
    array, axis=-1, *, highlevel=True, behavior=None, attrs=None
):
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
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Simplifies unions of records, e.g.

        >>> array = ak.concatenate(([{"a": 1}], [{"b": 2}]))
        >>> array
        <Array [{a: 1}, {b: 2}] type='2 * union[{a: int64}, {b: int64}]'>

    into records of options, i.e.

        >>> ak.merge_union_of_records(array)
        <Array [{a: 1, b: None}, {a: None, ...}] type='2 * {a: ?int64, b: ?int64}'>

    Missing records are preserved in the result, e.g.

        >>> array = ak.concatenate(([{"a": 1}], [{"b": 2}, None]))
        >>> array
        <Array [{a: 1}, {b: 2}, None] type='3 * union[{a: int64}, ?{b: int64}]'>
        >>> ak.merge_union_of_records(array)
        <Array [{a: 1, b: None}, {...}, None] type='3 * ?{a: ?int64, b: ?int64}'>
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, axis, highlevel, behavior, attrs)


def _impl(array, axis, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    def invert_record_union(
        tags: ArrayLike, index: ArrayLike, contents
    ) -> ak.contents.RecordArray:
        index_nplike = layout.backend.index_nplike
        # First, create an ordered list containing the union of all fields
        seen_fields = set()
        all_fields = []
        for content in contents:
            # Find new fields
            for field in content.fields:
                if field not in seen_fields:
                    seen_fields.add(field)
                    all_fields.append(field)

        # Build unions for each field
        outer_field_contents = []
        for field in all_fields:
            field_tags = index_nplike.asarray(tags.data, copy=True)
            field_index = index_nplike.asarray(index.data, copy=True)

            # Build contents for union representing current field
            field_contents = [c.content(field) for c in contents if c.has_field(field)]

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
            if len(field_contents) < len(contents):
                # Make the tagged content an option, growing by one to ensure we
                # have a known `None` value to index into
                tagged_content = field_contents[tag_for_missing]
                indexedoption_index = index_nplike.arange(
                    tagged_content.length + 1, dtype=np.int64
                )
                indexedoption_index[
                    index_nplike.shape_item_as_index(tagged_content.length)
                ] = -1
                field_contents[tag_for_missing] = (
                    ak.contents.IndexedOptionArray.simplified(
                        ak.index.Index64(indexedoption_index), tagged_content
                    )
                )

            # Index of None values in tagged content (content with extra None item at end)
            index_missing = index_nplike.shape_item_as_index(
                field_contents[tag_for_missing].length - 1
            )
            # Now build contents for union, by looping over outermost index
            # Overwrite tags to adjust for new contents length
            # and use the tagged content for any missing values
            k = 0
            for j, content in enumerate(contents):
                tag_is_j = field_tags == j

                if content.has_field(field):
                    # Rewrite tags to account for missing fields
                    field_tags[tag_is_j] = k
                    k += 1

                else:
                    # Rewrite tags to point to option content
                    field_tags[tag_is_j] = tag_for_missing
                    # Point each value to missing value
                    field_index[tag_is_j] = index_missing

            outer_field_contents.append(
                ak.contents.UnionArray.simplified(
                    ak.index.Index8(field_tags),
                    ak.index.Index64(field_index),
                    field_contents,
                )
            )
        return ak.contents.RecordArray(
            outer_field_contents, all_fields, backend=layout.backend
        )

    def compact_option_index(index: ArrayLike) -> ArrayLike:
        # Find dense (outer) index into non-null items.
        # This is in trivial order: the re-arranging is done by the union (below)
        is_none = index < 0
        num_none = layout.backend.index_nplike.count_nonzero(is_none)
        dense_index = layout.backend.index_nplike.empty(index.size, dtype=index.dtype)
        dense_index[is_none] = -1
        dense_index[~is_none] = layout.backend.index_nplike.arange(
            index.size - num_none,
            dtype=index.dtype,
        )
        return dense_index

    def apply(layout, depth, backend, **kwargs):
        posaxis = maybe_posaxis(layout, axis, depth)
        if posaxis is None:
            mindepth, maxdepth = layout.minmax_depth
            raise TypeError(
                f"cannot merge_union_of_records with axis={axis} if the array has multiple levels of nesting depth; this array has depths from {mindepth} to {maxdepth} (inclusive); try setting axis to a non-negative value"
            )

        if depth < posaxis + 1 and layout.is_leaf:
            raise AxisError(f"axis={axis} exceeds the depth of this array ({depth})")
        elif depth == posaxis + 1 and layout.is_union:
            if not all(
                x.is_record or x.is_indexed or x.is_option for x in layout.contents
            ):
                return

            # Any option types need to be re-written
            if any(x.is_option for x in layout.contents):
                # We'll rebuild the union to include only the non-null items.
                inner_union_index_parts = []
                next_contents = []
                next_tags_data_sparse = backend.index_nplike.asarray(
                    layout.tags.data, copy=True
                )
                for tag, content in enumerate(layout.contents):
                    is_this_tag = layout.tags.data == tag

                    # Union arrays for this content
                    tag_index_data = layout.index.data[is_this_tag]

                    # For unmasked arrays, we can directly take the content
                    if isinstance(content, ak.contents.UnmaskedArray):
                        next_contents.append(content.content)
                        inner_union_index_parts.append(tag_index_data)
                    # Otherwise, we need to rebuild the index
                    elif content.is_option or content.is_indexed:
                        # Let's work with indexed option types for ease
                        if content.is_option:
                            content = content.to_IndexedOptionArray64()

                        # First, find the inner index that actually re-arranges the (non-null) items
                        merged_index = content.index.data[tag_index_data]
                        is_non_null = merged_index >= 0
                        inner_union_index_parts.append(merged_index[is_non_null])
                        # Mask out tags of items that are missing
                        next_tags_data_sparse[is_this_tag] = backend.index_nplike.where(
                            is_non_null, tag, -1
                        )

                        # The length of this index/option content is irrelevant; the union provides the length
                        next_contents.append(content.content)
                    # Non-indexed/option types are trivially included as-is
                    else:
                        next_contents.append(content)
                        inner_union_index_parts.append(tag_index_data)

                # We'll create an outermost indexed-option type, which re-instates the missing values.
                # This should have the same length as the original union, and its index should be "dense"
                # (contiguous, monotonic integers; or -1). Therefore, we can directly compute it from the "sparse"
                # tags index, which has the same length as the original union, and has only missing items set to -1.
                outer_option_dense_index = compact_option_index(next_tags_data_sparse)

                # Ignore missing items for inner union, creating a dense array of tags
                next_tags_data = next_tags_data_sparse[next_tags_data_sparse >= 0]
                # Build dense index from parts for each tag
                next_index_data = backend.index_nplike.empty(
                    next_tags_data.size, dtype=np.int64
                )
                for tag, content_index in enumerate(inner_union_index_parts):
                    next_index_data[next_tags_data == tag] = content_index

                # Return option around record of unions
                return ak.contents.IndexedOptionArray(
                    ak.index.Index64(outer_option_dense_index),
                    invert_record_union(next_tags_data, next_index_data, next_contents),
                )

            # Any index types need to be re-written
            elif any(x.is_indexed for x in layout.contents):
                # We'll create an outermost indexed-option type, which re-instates the missing values
                next_index_data = backend.index_nplike.empty(
                    layout.index.length, dtype=np.int64
                )

                # We'll rebuild the union to include only the non-null items.
                next_contents = []
                for tag, content in enumerate(layout.contents):
                    is_this_tag = layout.tags.data == tag

                    # Rewrite union index of indexed types
                    if content.is_indexed:
                        next_index_data[is_this_tag] = content.index.data[
                            content.index.data[is_this_tag]
                        ]
                        next_contents.append(content.content)

                    else:
                        next_index_data[is_this_tag] = content.index.data[is_this_tag]
                        next_contents.append(content)

                return invert_record_union(
                    layout.tags.data, next_index_data, next_contents
                )

            else:
                return invert_record_union(
                    layout.tags.data, layout.index.data, layout.contents
                )

    out = ak._do.recursively_apply(layout, apply)
    return ctx.wrap(out, highlevel=highlevel)
