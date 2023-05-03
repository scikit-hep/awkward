from __future__ import annotations

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
# ruff: noqa: B023
__all__ = ("enforce_type",)


from enum import Enum
from itertools import permutations

import awkward as ak
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._parameters import type_parameters_equal
from awkward.types.numpytype import primitive_to_dtype

np = NumpyMetadata.instance()


def _layout_has_type(layout: ak.contents.Content, type_: ak.types.Type) -> bool:
    """
    Args:
        layout: content object
        type_: low-level type object

    Returns True if the layout satisfies the given type;, otherwise False.
    """
    if not type_parameters_equal(layout._parameters, type_._parameters):
        return False

    if layout.is_unknown:
        return isinstance(type_, ak.types.UnknownType)
    elif layout.is_option:
        return isinstance(type_, ak.types.OptionType) and _layout_has_type(
            layout.content, type_.content
        )
    elif layout.is_indexed:
        return _layout_has_type(layout.content, type_)
    elif layout.is_regular:
        return (
            isinstance(type_, ak.types.RegularType)
            and (
                layout.size is unknown_length
                or type_.size is unknown_length
                or layout.size == type_.size
            )
            and _layout_has_type(layout.content, type_.content)
        )
    elif layout.is_list:
        return isinstance(type_, ak.types.ListType) and _layout_has_type(
            layout.content, type_.content
        )
    elif layout.is_numpy:
        for _ in range(layout.purelist_depth - 1):
            if not isinstance(type_, ak.types.RegularType):
                return False
            type_ = type_.content
        return isinstance(
            type_, ak.types.NumpyType
        ) and layout.dtype == primitive_to_dtype(type_.primitive)
    elif layout.is_record:
        if (
            not isinstance(type_, ak.types.RecordType)
            or type_.is_tuple != layout.is_tuple
        ):
            return False

        if layout.is_tuple:
            return all(
                _layout_has_type(c, t) for c, t in zip(layout.contents, type_.contents)
            )
        else:
            return (frozenset(layout.fields) == frozenset(type_.fields)) and all(
                _layout_has_type(layout.content(f), type_.content(f))
                for f in type_.fields
            )
    elif layout.is_union:
        if len(layout.contents) != len(type_.contents):
            return False

        for contents in permutations(layout.contents):
            if all(
                _layout_has_type(layout, type_)
                for layout, type_ in zip(contents, type_.contents)
            ):
                return True
        return False
    else:
        raise TypeError(layout)


class UnionErasureMode(str, Enum):
    PROJECT = "project"
    CONVERT = "convert"


def enforce_type(
    array,
    type,
    *,
    union_erasure=UnionErasureMode.CONVERT,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        type (#ak.types.Type or str): The type that `array` will be enforced to.
        union_erasure (str): Rule for erasing unions, one of:
            - `"convert"` - convert all union contents to the given type
            - `"project"` - project out the first content with exactly the given type
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns an array whose structure is modified to match the given type.

    In addition to preserving the existing type and/or changing parameters,

    - #ak.types.OptionType can be added or removed (if there are no missing values)
    - #ak.types.UnionType can

      * grow to include new variant types,
      * project to a single type (if the union contains no values for this type),
      * convert to a single type,
      * change type in a single variant.
      Due to these rules, changes to more than one variant of a union must be performed with multiple calls to #ak.enforce_type
    - #ak.types.RecordType can

      * grow to include new fields / slots,
      * shrink to drop existing fields / slots.

      A #ak.types.RecordType may only be converted to another #ak.types.RecordType if it is of the same flavour, i.e.
      tuples can be converted to tuples, or records to records. Where a new field/slot is added to a #ak.types.RecordType,
      it must be an #ak.types.OptionType. For tuples, slots may only be added to the end of the tuple
    - #ak.types.ListType can convert to a #ak.types.RegularType
    - #ak.types.NumpyType can change primitive
    - #ak.types.UnknownType can be converted to any other type, and be converted to from any other type.
    The conversion rules outlined above are not data-dependent; the appropriate rule is chosen from the layout and the
    given type value. If the conversion is not possible given the layout data, e.g. a conversion from an irregular list
    to a regular type, it will fail.
    """
    with ak._errors.OperationErrorContext(
        "ak.enforce_type",
        {
            "array": array,
            "type": type,
            "highlevel": highlevel,
            "behavior": behavior,
            "union_erasure": union_erasure,
        },
    ):
        return _impl(array, type, union_erasure, highlevel, behavior)


def _option_to_projected_indexed_option(
    layout: ak.contents.IndexedOptionArray
    | ak.contents.BitMaskedArray
    | ak.contents.ByteMaskedArray
    | ak.contents.UnmaskedArray,
) -> ak.contents.IndexedOptionArray:
    """
    Args:
        layout: option-type layout

    Returns a new IndexedOptionArray whose contents are already projected.
    """
    index_nplike = layout.backend.index_nplike
    new_index = index_nplike.empty(layout.length, dtype=np.int64)

    is_none = layout.mask_as_bool(False)
    num_none = index_nplike.count_nonzero(is_none)

    new_index[is_none] = -1
    new_index[~is_none] = index_nplike.arange(
        layout.length - num_none,
        dtype=new_index.dtype,
    )
    return ak.contents.IndexedOptionArray(
        ak.index.Index64(new_index, nplike=index_nplike),
        layout.project(),
        parameters=layout._parameters,
    )


def _drop_option(
    layout: ak.contents.IndexedOptionArray
    | ak.contents.BitMaskedArray
    | ak.contents.ByteMaskedArray
    | ak.contents.UnmaskedArray,
    lazy: bool,
) -> ak.contents.IndexedArray:
    """
    Args:
        layout: option-type layout
        lazy: whether to keep indirection via an IndexedArray

    Returns a new Content containing the non-missing items of `layout`.
    This is equivalent to dropping the None values.
    Preserve an indirection if `lazy` is True, and the option is indexed
    """
    if not layout.is_indexed:
        return layout.content[: layout.length]
    elif lazy:
        # Convert option to IndexedOptionArray and determine index of valid values
        layout = layout.to_IndexedOptionArray64()
        return ak.contents.IndexedArray.simplified(
            index=ak.index.Index64(layout.index.data[layout.mask_as_bool(True)]),
            content=layout.content,
            parameters=layout._parameters,
        )
    else:
        return layout.project()


def _recurse_indexed_any(
    layout: ak.contents.IndexedArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    if _layout_has_type(layout, type_):
        # If the types match, then we don't need to project, as only parameters
        # are changed (if at all)
        return layout.copy(content=_recurse(layout.content, type_, union_erasure))
    else:
        # Otherwise, to ensure that we can project out options, we need to know
        # exactly what's visible to the user
        return _recurse(layout.project(), type_, union_erasure)


def _recurse_unknown_any(
    layout: ak.contents.EmptyArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    type_form = ak.forms.from_type(type_)

    return type_form.length_zero_array(highlevel=False).copy(
        parameters=type_.parameters
    )


def _recurse_option_any(
    layout: ak.contents.IndexedOptionArray
    | ak.contents.BitMaskedArray
    | ak.contents.ByteMaskedArray
    | ak.contents.UnmaskedArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    # option → option (no change)
    if isinstance(type_, ak.types.OptionType):
        # Optimisation: is the layout changed below this level?
        if not _layout_has_type(layout.content, type_.content):
            # If so, convert to packed so that any non-referenced content items are trimmed
            # This is required so that unused union items are seen to be safe to project out later
            # We don't use to_packed(), as it recurses
            layout = _option_to_projected_indexed_option(layout)

        return layout.copy(
            content=_recurse(layout.content, type_.content, union_erasure),
            parameters=type_.parameters,
        )

    # drop option!
    else:
        if layout.backend.index_nplike.any(layout.mask_as_bool(False)):
            raise ValueError(
                "option types can only be removed if there are no missing values"
            )
        else:
            # If so, drop the option-type, but don't project anything
            return _recurse(
                _drop_option(layout, lazy=_layout_has_type(layout.content, type_)),
                type_,
                union_erasure,
            )


def _recurse_any_option(
    layout: ak.contents.Content,
    type_: ak.types.OptionType,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    return ak.contents.UnmaskedArray(
        _recurse(layout, type_.content, union_erasure), parameters=type_.parameters
    )


def _recurse_union_any(
    layout: ak.contents.UnionArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    # If the target is a union type, then we have to determine the solution for e.g.
    # {A, B, C, D} → {X, Y, C, Z}.
    if isinstance(type_, ak.types.UnionType):
        return _recurse_union_union(layout, type_, union_erasure)
    # Otherwise, we are projecting out the union to a single type
    else:
        return _recurse_union_non_union(layout, type_, union_erasure)


def _recurse_union_union(
    layout: ak.contents.UnionArray,
    type_: ak.types.UnionType,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    n_type_contents = len(type_.contents)
    n_layout_contents = len(layout.contents)
    # The general operation of converting between one union and another can be decomposed into multiple
    # conversions from {A, B, C} to {A, B}, or from {A, B} to {A, B, C}.
    # Here we will only allow these transformations, because (a) the code is easier to write!
    # and (b) it is much easier to reason about _as a user_.

    # If the target has _more_ contents, then we assume we can add an unused content
    # Assume here that we have a *subset* of the type, i.e layout is {A, B, C}
    # and type is {A, B, C, D, ...}.
    if n_type_contents > n_layout_contents:
        # We can permute the type order, as union contents are not ordered
        # Permute the index, so that we can later recover remaining types
        ix_contents = range(n_type_contents)
        for ix_perm_contents in permutations(ix_contents, n_layout_contents):
            retained_types = [type_.contents[j] for j in ix_perm_contents]
            # Require that all layouts match types for layout permutation
            if not all(
                _layout_has_type(c, t) for c, t in zip(layout.contents, retained_types)
            ):
                continue

            missing_types = [
                type_.contents[j]
                for j in (frozenset(ix_contents) - frozenset(ix_perm_contents))
            ]

            # We only need to recurse here to enable parameter changes
            # Given that we _know_ all layouts match their types for the permutation,
            # we don't need to project these contents — they won't be operated upon (besides parameters)
            contents = [
                _recurse(c, t, union_erasure)
                for c, t in zip(layout.contents, retained_types)
            ]
            contents.extend(
                [
                    ak.forms.from_type(t).length_zero_array(
                        highlevel=False, backend=layout.backend
                    )
                    for t in missing_types
                ]
            )
            return layout.copy(contents=contents, parameters=type_.parameters)

        # No permutation succeeded
        raise NotImplementedError(
            "UnionArray(s) can currently only be converted into UnionArray(s) with a greater number contents if the "
            "layout contents are equal to some permutation of the type contents "
        )

    # Otherwise, we assume that we're projecting out one (or more) of our contents
    # Assume here that we have a *subset* of the layout, i.e layout is {A, B, C, D, ...}
    # and type is {A, B, C}. As the layout needs to lose a content, we must hope that the matching
    # permutation (by type) is also one that drops only unused contents from the union,
    # as layout operation must be typetracer-predictable
    elif n_layout_contents > n_type_contents:
        ix_contents = range(n_layout_contents)
        for ix_perm_contents in permutations(ix_contents, n_type_contents):
            retained_contents = [layout.contents[j] for j in ix_perm_contents]
            # Require that all layouts match types for layout permutation
            if not all(
                _layout_has_type(c, t)
                for c, t in zip(retained_contents, type_.contents)
            ):
                continue

            is_trivial_permutation = ix_perm_contents == range(n_type_contents)
            # Optimisation: if this is the trivial permutation, swe don't need to do any tag re-arranging
            if is_trivial_permutation:
                layout_tags = layout.tags
            else:
                layout_tags = ak.index.Index8.empty(
                    layout.tags.length, layout.backend.index_nplike
                )

            # Ensure that the union references all of the tags of the permutation,
            # and re-order the tags if this is not the trivial permutation
            _total_used_tags = 0
            for i, j in zip(ix_perm_contents, range(n_type_contents)):
                layout_tag_is_i = layout.tags.data == i

                # Rewrite the tags if they need to be condensed (i.e., not if this is the trivial permutation)
                if not is_trivial_permutation:
                    layout_tags.data[layout_tag_is_i] = j

                # Keep track of the length of layout subcontent
                _total_used_tags += layout.backend.index_nplike.count_nonzero(
                    layout_tag_is_i
                )
            # Is the new union of the same length as the original?
            total_used_tags = layout.backend.index_nplike.index_as_shape_item(
                _total_used_tags
            )
            if not (
                total_used_tags is unknown_length
                or layout.length is unknown_length
                or total_used_tags == layout.length
            ):
                raise ValueError("union conversion must not be lossless")

            return layout.copy(
                tags=layout_tags,
                # We only need to recurse here to enable parameter changes
                # Given that we _know_ all layouts match their types for the permutation,
                # we don't need to project these contents — they won't be operated upon (besides parameters)
                contents=[
                    _recurse(c, t, union_erasure)
                    for c, t in zip(retained_contents, type_.contents)
                ],
                parameters=type_.parameters,
            )

        # TODO: add note about expand and contract
        raise NotImplementedError(
            "UnionArray(s) can currently only be converted into UnionArray(s) with a greater "
            "number of contents if the layout contents are compatible with some permutation of "
            "the type contents"
        )

    # Type and layout have same number of contents. Up-to *one* content can differ
    else:
        ix_contents = range(n_type_contents)
        for ix_perm_contents in permutations(ix_contents):
            permuted_types = [type_.contents[j] for j in ix_perm_contents]

            # How many contents match types in this permutation?
            content_matches_type = [
                _layout_has_type(c, t) for c, t in zip(layout.contents, permuted_types)
            ]
            n_matching = sum(content_matches_type, 0)

            # If all contents are nominally equal to the position-matched type, then only parameters have changed
            if n_matching == len(type_.contents):
                return layout.copy(
                    contents=[
                        _recurse(c, t, union_erasure)
                        for c, t in zip(layout.contents, permuted_types)
                    ],
                    parameters=type_.parameters,
                )
            # Single content differs, we can convert by position
            elif n_matching == len(type_.contents) - 1:
                next_contents = []
                index = layout.index
                for tag, content_type, is_match in zip(
                    range(len(layout.contents)), permuted_types, content_matches_type
                ):
                    # If the types agree between the intended type and content, then include this content
                    # as-is, only recursing to update parameters. Because the types agree, we're safe
                    # not to project out this content
                    # TODO: don't walk into the tree if nothing needs doing.
                    if is_match:
                        next_contents.append(
                            _recurse(layout.contents[tag], content_type, union_erasure)
                        )
                    # Otherwise, this content is being converted, and we need to recurse into the projection
                    # to ensure that the content is packed
                    else:
                        layout_content = layout.project(tag)
                        next_contents.append(
                            _recurse(layout_content, content_type, union_erasure)
                        )

                        # Rebuild the index as an enumeration over the (dense) projection
                        # This ensures that it is packed!
                        index_data = layout.backend.index_nplike.asarray(
                            index.data, copy=True
                        )
                        is_tag = layout.tags.data == tag
                        index_data[is_tag] = layout.backend.index_nplike.arange(
                            layout_content.length, dtype=index_data.dtype
                        )
                        index = ak.index.Index(index_data)

                return layout.copy(
                    index=index,
                    contents=next_contents,
                    parameters=type_.parameters,
                )
            else:
                raise TypeError(
                    "UnionArray(s) can currently only be converted into UnionArray(s) with the same number of contents "
                    "if no greater than one content differs in type"
                )


def _recurse_union_non_union(
    layout: ak.contents.UnionArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    index_nplike = layout.backend.index_nplike

    if union_erasure == UnionErasureMode.PROJECT:
        # Find the first content whose type equals the given type
        for tag, content in enumerate(layout.contents):  # noqa: B007
            if _layout_has_type(content, type_):
                break
        else:
            raise TypeError(
                f"UnionArray(s) can only be converted into {type_} if it is compatible, but no "
                "compatible content as found"
            )

        # Require that we are the only content
        content_is_tag = layout.tags.data == tag
        if index_nplike.known_data and not index_nplike.all(content_is_tag):
            raise ValueError(
                f"UnionArray(s) can only be converted to {type_} if they are equivalent to their "
                f"projections"
            )
        else:
            # We don't need to pack, as the type hasn't changed, so introduce an indexed type.
            # This ensures that unions over records don't needlessly project.
            # From the canonical rules, the content of a union *can* be an index, so we use simplified
            return ak.contents.IndexedArray.simplified(
                ak.index.Index(layout.index.data[content_is_tag]),
                _recurse(content, type_, union_erasure),
            )
    else:
        # Convert each projected content to the required type
        next_contents = []
        index_data = index_nplike.empty(layout.length, dtype=np.int64)
        j = 0
        for tag in range(len(layout.contents)):
            tag_content = layout.project(tag)
            # Set the index of these tags to a simple range
            i, j = j, j + index_nplike.shape_item_as_index(tag_content.length)
            index_data[layout.tags.data == tag] = index_nplike.arange(
                i, j, dtype=np.int64
            )
            # Convert layout
            next_contents.append(enforce_type(tag_content, type_))

        # Merge the results
        for content in next_contents[1:]:
            assert ak._do.mergeable(next_contents[0], content, mergebool=False)
        next_content = ak._do.mergemany(next_contents)

        # Index over them
        index = ak.index.Index64(index_data)
        return ak.contents.IndexedArray(index, next_content)


def _recurse_any_union(
    layout: ak.contents.Content,
    type_: ak.types.UnionType,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    for i, content_type in enumerate(type_.contents):
        if not _layout_has_type(layout, content_type):
            continue

        tags = layout.backend.index_nplike.zeros(layout.length, dtype=np.int8)
        index = layout.backend.index_nplike.arange(layout.length, dtype=np.int64)

        other_contents = [
            ak.forms.from_type(t).length_zero_array(
                backend=layout.backend, highlevel=False
            )
            for j, t in enumerate(type_.contents)
            if j != i
        ]

        return ak.contents.UnionArray(
            tags=ak.index.Index8(tags, nplike=layout.backend.index_nplike),
            index=ak.index.Index64(index, nplike=layout.backend.index_nplike),
            contents=[_recurse(layout, content_type, union_erasure), *other_contents],
            parameters=type_.parameters,
        )

    raise TypeError(
        f"{type(layout).__name__} can only be converted into a UnionType if it is compatible with one "
        "of its contents, but no compatible content as found"
    )


def _recurse_regular_any(
    layout: ak.contents.RegularArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    if isinstance(type_, ak.types.RegularType):
        # regular → regular requires same size!
        if layout.size != type_.size:
            raise ValueError(
                f"regular layout has different size ({layout.size}) to type ({type_.size})"
            )

        return layout.copy(
            content=_recurse(layout.content, type_.content, union_erasure),
            parameters=type_.parameters,
        )

    elif isinstance(type_, ak.types.ListType):
        layout_list = layout.to_ListOffsetArray64(True)
        return layout_list.copy(
            content=_recurse(layout_list.content, type_.content, union_erasure),
            parameters=type_.parameters,
        )

    else:
        raise TypeError(
            f"lists can only be converted to lists, options of lists, or unions thereof, not {type_}"
        )


def _recurse_list_any(
    layout: ak.contents.ListArray | ak.contents.ListOffsetArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    if isinstance(type_, ak.types.RegularType):
        layout_regular = layout.to_RegularArray()
        if layout_regular.size != type_.size:
            raise ValueError(
                f"converted regular layout has different size ({layout_regular.size}) to type ({type_.size})"
            )

        return layout_regular.copy(
            # The result of `to_RegularArray` should already be packed
            content=_recurse(layout_regular.content, type_.content, union_erasure),
            parameters=type_.parameters,
        )

    elif isinstance(type_, ak.types.ListType):
        if _layout_has_type(layout.content, type_.content):
            # Don't need to pack the content
            return layout.copy(
                content=_recurse(layout.content, type_.content, union_erasure),
                parameters=type_.parameters,
            )
        else:
            # Need to pack the content!
            layout = layout.to_ListOffsetArray64(True)
            layout = layout[: layout.offsets[-1]]
            return layout.copy(
                content=_recurse(layout.content, type_.content, union_erasure),
                parameters=type_.parameters,
            )

    else:
        raise TypeError(
            f"lists can only be converted to lists, options of lists, or unions thereof, not {type_}"
        )


def _recurse_numpy_any(
    layout: ak.contents.NumpyArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    if len(layout.shape) == 1:
        if not isinstance(type_, ak.types.NumpyType):
            raise TypeError(
                "NumpyArray(s) can only be converted into NumpyArray(s), options of NumpyArray(s), or "
                "unions thereof"
            )
        return ak.values_astype(
            # HACK: drop parameters from type so that character arrays are supported
            layout.copy(parameters=None),
            to=primitive_to_dtype(type_.primitive),
            highlevel=False,
        ).copy(parameters=type_.parameters)

    else:
        assert len(layout.shape) > 0
        return _recurse(layout.to_RegularArray(), type_, union_erasure)


def _recurse_record_any(
    layout: ak.contents.RecordArray,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    if isinstance(type_, ak.types.RecordType):
        if type_.is_tuple and layout.is_tuple:
            # Recurse into shared contents
            type_contents = iter(type_.contents)
            next_contents = [
                _recurse(c, t, union_erasure)
                for c, t in zip(layout.contents, type_contents)
            ]
            # Anything left in `type_contents` are the types of new slots
            for next_type in type_contents:
                if not isinstance(next_type, ak.types.OptionType):
                    raise TypeError(
                        "can only add new slots to a tuple if they are option types"
                    )
                # Append new contents
                next_contents.append(
                    ak.contents.IndexedOptionArray.simplified(
                        ak.index.Index64(
                            layout.backend.index_nplike.full(layout.length, -1)
                        ),
                        ak.forms.from_type(next_type).length_zero_array(
                            backend=layout.backend, highlevel=False
                        ),
                    )
                )

            return layout.copy(
                fields=None,
                contents=next_contents,
                parameters=type_.parameters,
            )

        elif not (type_.is_tuple or layout.is_tuple):
            type_fields = frozenset(type_.fields)
            layout_fields = frozenset(layout._fields)

            # Compute existing and new fields
            existing_fields = list(type_fields & layout_fields)
            new_fields = list(type_fields - layout_fields)
            next_fields = existing_fields + new_fields

            # Recurse into shared contents
            next_contents = [
                _recurse(layout.content(f), type_.content(f), union_erasure)
                for f in existing_fields
            ]
            for field in new_fields:
                field_type = type_.content(field)
                if not isinstance(field_type, ak.types.OptionType):
                    raise TypeError(
                        "can only add new slots to a tuple if they are option types"
                    )
                # Append new contents
                next_contents.append(
                    ak.contents.IndexedOptionArray.simplified(
                        ak.index.Index64(
                            layout.backend.index_nplike.full(layout.length, -1)
                        ),
                        ak.forms.from_type(field_type).length_zero_array(
                            backend=layout.backend, highlevel=False
                        ),
                    )
                )

            return layout.copy(
                fields=next_fields,
                contents=next_contents,
                parameters=type_.parameters,
            )

        else:
            raise ValueError(
                "RecordArray(s) cannot be converted between records and tuples."
            )
    else:
        raise TypeError(
            "RecordArray(s) can only be converted into RecordArray(s), options of RecordArray(s), or "
            "unions thereof"
        )


# Require parameters be conserved
def _recurse(
    layout: ak.contents.Content,
    type_: ak.types.Type,
    union_erasure: UnionErasureMode,
) -> ak.contents.Content:
    """
    Args:
        layout: layout to recurse into
        type_: expected type of the recursion result

    Returns a callable which converts from `layout` to `type_`. layout ensures that
    calling `recurse` is a type-only program, whilst keeping the conversion logic
    adjacent to the type check logic.
    """

    if layout.is_unknown:
        return _recurse_unknown_any(layout, type_, union_erasure)

    elif layout.is_option:
        return _recurse_option_any(layout, type_, union_erasure)

    elif layout.is_indexed:
        return _recurse_indexed_any(layout, type_, union_erasure)

    # Here we *don't* have any layouts that are options, unknowns, or indexed
    # If we see an option, we are therefore *adding* one
    elif isinstance(type_, ak.types.OptionType):
        return _recurse_any_option(layout, type_, union_erasure)

    elif layout.is_union:
        return _recurse_union_any(layout, type_, union_erasure)

    # Here we *don't* have any layouts that are options, unknowns, indexed, or unions
    # If we see a union, we are therefore *adding* one
    elif isinstance(type_, ak.types.UnionType):
        return _recurse_any_union(layout, type_, union_erasure)

    elif layout.is_regular:
        return _recurse_regular_any(layout, type_, union_erasure)

    elif layout.is_list:
        return _recurse_list_any(layout, type_, union_erasure)

    elif layout.is_numpy:
        return _recurse_numpy_any(layout, type_, union_erasure)

    elif layout.is_record:
        return _recurse_record_any(layout, type_, union_erasure)
    else:
        raise NotImplementedError(type(layout), type_)


def _impl(array, type_, union_erasure, highlevel, behavior):
    if union_erasure not in {UnionErasureMode.PROJECT, UnionErasureMode.CONVERT}:
        raise ValueError(union_erasure)

    layout = ak.to_layout(array)

    if isinstance(type_, str):
        type_ = ak.types.from_datashape(type_, highlevel=False)

    out = _recurse(layout, type_, union_erasure)
    return wrap_layout(out, like=array, behavior=behavior, highlevel=highlevel)
