# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
# ruff: noqa: B023
__all__ = ("enforce_type",)

from itertools import permutations

import awkward as ak
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._typing import Callable, TypeAlias
from awkward.types.numpytype import primitive_to_dtype

np = NumpyMetadata.instance()


BuilderType: TypeAlias = "Callable[[ak.contents.Content], ak.contents.Content]"
ErrorHandlerType: TypeAlias = "Callable[[Exception], None]"


def enforce_type(
    array,
    type,
    *,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        type (#ak.types.Type or str): The type of the Awkward
            Array to enforce.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.


    """
    with ak._errors.OperationErrorContext(
        "ak.enforce_type",
        {
            "array": array,
            "type": type,
            "highlevel": highlevel,
            "behavior": behavior,
        },
    ):
        return _impl(array, type, highlevel, behavior)


class InvalidTypeConversionError(RuntimeError):
    """Internal error raised for control flow"""


def raise_error(err: Exception):
    raise err


def raise_invalid_type_conversion_error(err: Exception):
    """
    Args:
        err: true exception

    Raises an `InvalidTypeConversionError` in place of the true error, so that a simple error handler can be used.

    """
    raise InvalidTypeConversionError from err


def recurse_indexed_any(
    layout: ak.contents.IndexedArray,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    build_subtree = recurse(layout.content, type_, handle_error)

    def thunk(this):
        return build_subtree(this.project())

    return thunk


def recurse_unknown_any(
    layout: ak.contents.EmptyArray,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    type_form = ak.forms.from_type(type_)

    def thunk(this):
        return type_form.length_zero_array(highlevel=False).copy(
            parameters=type_.parameters
        )

    return thunk


def recurse_option_any(
    layout: ak.contents.Content,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    # option → option (no change)
    if isinstance(type_, ak.types.OptionType):
        build_subtree = recurse(layout.content, type_.content, handle_error)

        def thunk(this):
            return this.copy(
                content=build_subtree(this.content), parameters=type_.parameters
            )

        return thunk
    # drop option!
    else:
        build_subtree = recurse(layout.content, type_, handle_error)

        def thunk(this):
            if this.backend.index_nplike.all(this.mask_as_bool(True)):
                this_no_option = ak.drop_none(this, axis=0, highlevel=False)
                return build_subtree(this_no_option)
            else:
                raise ValueError(
                    "option types can only be removed if there are no missing values"
                )

        return thunk


def recurse_any_option(
    layout: ak.contents.Content,
    type_: ak.types.OptionType,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    build_subtree = recurse(layout, type_.content, handle_error)

    def thunk(this):
        return ak.contents.UnmaskedArray(
            build_subtree(this), parameters=type_.parameters
        )

    return thunk


def recurse_union_any(
    layout: ak.contents.UnionArray,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    # If the target is a union type, then we have to determine the solution for e.g.
    # {A, B, C, D} → {X, Y, C, Z}.
    if isinstance(type_, ak.types.UnionType):
        return recurse_union_union(layout, type_, handle_error)
    # Otherwise, we are projecting out the union to a single type
    else:
        return recurse_union_non_union(layout, type_, handle_error)


def recurse_union_union(
    layout: ak.contents.UnionArray,
    type_: ak.types.UnionType,
    handle_error: ErrorHandlerType,
) -> BuilderType:
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
            try:
                builders = [
                    recurse(
                        c,
                        type_.contents[j],
                        raise_invalid_type_conversion_error,
                    )
                    for c, j in zip(layout.contents, ix_perm_contents)
                ]
            except InvalidTypeConversionError:
                continue
            else:

                def thunk(this):
                    ix_missing_contents = frozenset(ix_contents) - frozenset(
                        ix_perm_contents
                    )
                    missing_types = [type_.contents[j] for j in ix_missing_contents]
                    contents = [b(c) for b, c in zip(builders, this.contents)]
                    contents.extend(
                        [
                            ak.forms.from_type(t).length_zero_array(
                                highlevel=False, backend=this.backend
                            )
                            for t in missing_types
                        ]
                    )
                    return this.copy(contents=contents, parameters=type_.parameters)

                return thunk
        # No permutation succeeded
        handle_error(
            NotImplementedError(
                "UnionArray(s) can currently only be converted into UnionArray(s) with a greater number contents if the "
                "layout contents are compatible with some permutation of the type contents "
            )
        )

    # Otherwise, we assume that we're projecting out one (or more) of our contents
    # Assume here that we have a *subset* of the layout, i.e layout is {A, B, C, D, ...}
    # and type is {A, B, C}. As the layout needs to lose a content, we must hope that the matching
    # permutation (by type) is also one that drops only unused contents from the union,
    # as this operation must be typetracer-predictable
    else:
        ix_contents = range(n_layout_contents)
        for ix_perm_contents in permutations(ix_contents, n_type_contents):
            try:
                builders = [
                    recurse(
                        layout.contents[j],
                        t,
                        raise_invalid_type_conversion_error,
                    )
                    for j, t in zip(ix_perm_contents, type_.contents)
                ]
            except InvalidTypeConversionError:
                continue
            else:

                def thunk(this):
                    is_trivial_permutation = ix_perm_contents == range(n_type_contents)
                    if is_trivial_permutation:
                        # The trivial permutation won't require any copying of tags
                        this_tags = this.tags
                    else:
                        this_tags = ak.index.Index8.empty(
                            this.tags.length, this.backend.index_nplike
                        )

                    _total_used_tags = 0
                    this_contents = []

                    for i, j in zip(ix_perm_contents, range(n_type_contents)):
                        this_contents.append(this.contents[i])
                        this_tag_is_i = this.tags.data == i

                        # Rewrite the tags if they need to be condensed
                        if not is_trivial_permutation:
                            this_tags.data[this_tag_is_i] = j

                        # Keep track of the length of this subcontent
                        _total_used_tags += this.backend.index_nplike.count_nonzero(
                            this_tag_is_i
                        )
                    # Is the new union of the same length as the original?
                    total_used_tags = this.backend.index_nplike.index_as_shape_item(
                        _total_used_tags
                    )
                    if not (
                        total_used_tags is unknown_length
                        or this.length is unknown_length
                        or total_used_tags == this.length
                    ):
                        raise ValueError("union conversion must not be lossless")

                    return this.copy(
                        tags=this_tags,
                        contents=[b(c) for b, c in zip(builders, this_contents)],
                        parameters=type_.parameters,
                    )

                return thunk

        handle_error(
            # Add note about expand + contract
            NotImplementedError(
                "UnionArray(s) can currently only be converted into UnionArray(s) with a greater "
                "number of contents if the layout contents are compatible with some permutation of "
                "the type contents"
            )
        )


def recurse_union_non_union(
    layout: ak.contents.UnionArray,
    type_: ak.types.UnionType,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    for i, content in enumerate(layout.contents):
        try:
            builder = recurse(content, type_, raise_invalid_type_conversion_error)
        except InvalidTypeConversionError:
            continue
        else:

            def thunk(this):
                projected = this.project(i)
                if projected.length != this.length:
                    raise ValueError(
                        f"UnionArray(s) can only be converted to {type_} if they are equivalent to their "
                        f"projections"
                    )
                return builder(projected)

            return thunk
    handle_error(
        ValueError(
            f"UnionArray(s) can only be converted into {type_} if it is compatible, but no "
            "compatible content as found"
        )
    )


def recurse_any_union(
    layout: ak.contents.Content,
    type_: ak.types.UnionType,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    for i, content_type in enumerate(type_.contents):
        try:
            content_builder = recurse(
                layout, content_type, raise_invalid_type_conversion_error
            )
        except InvalidTypeConversionError:
            continue
        else:

            def thunk(this):
                tags = this.backend.index_nplike.zeros(this.length, dtype=np.int8)
                index = this.backend.index_nplike.arange(this.length, dtype=np.int64)

                other_contents = [
                    ak.forms.from_type(t).length_zero_array(
                        backend=this.backend, highlevel=False
                    )
                    for j, t in enumerate(type_.contents)
                    if j != i
                ]

                return ak.contents.UnionArray(
                    tags=ak.index.Index8(tags, nplike=this.backend.index_nplike),
                    index=ak.index.Index64(index, nplike=this.backend.index_nplike),
                    contents=[content_builder(this), *other_contents],
                    parameters=type_.parameters,
                )

            return thunk

    handle_error(
        ValueError(
            f"{type(layout).__name__} can only be converted into a UnionType if it is compatible with one "
            "of its contents, but no compatible content as found"
        )
    )


def recurse_list_1d(
    layout: ak.contents.Content,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    # These are "special" layout, so we need an exact (nominally) matching type
    if layout.form.type.is_equal_to(type_, all_parameters=False):

        def thunk(this):
            return this.copy(parameters=type_.parameters)

        return thunk
    else:
        handle_error(ValueError("form type does not match list type"))


def recurse_list_or_regular_any(
    layout: ak.contents.Content,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    if isinstance(type_, ak.types.RegularType):
        # regular → regular requires same size!
        if layout.is_regular and layout.size != type_.size:
            handle_error(
                ValueError(
                    f"regular layout has different size ({layout.size}) to type ({type_.size})"
                )
            )

        build_subtree = recurse(layout.content, type_.content, handle_error)

        def thunk(this):
            this_regular = this.to_RegularArray()
            if this_regular.size != type_.size:
                raise ValueError(
                    f"converted regular layout has different size ({this_regular.size}) to type ({type_.size})"
                )

            return this_regular.copy(
                content=build_subtree(this_regular.content),
                parameters=type_.parameters,
            )

        return thunk

    elif isinstance(type_, ak.types.ListType):
        build_subtree = recurse(layout.content, type_.content, handle_error)

        def thunk(this):
            this_list = this.to_ListOffsetArray64()
            return this_list.copy(
                content=build_subtree(this_list.content),
                parameters=type_.parameters,
            )

        return thunk
    else:
        handle_error(
            ValueError(
                f"lists can only be converted to lists, options of lists, or unions thereof, not {type_}"
            )
        )


def recurse_numpy_any(
    layout: ak.contents.NumpyArray,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    if len(layout.shape) == 1:
        if not isinstance(type_, ak.types.NumpyType):
            handle_error(
                ValueError(
                    "NumpyArray(s) can only be converted into NumpyArray(s), options of NumpyArray(s), or "
                    "unions thereof"
                )
            )

        def thunk(this):
            return ak.values_astype(
                this, to=primitive_to_dtype(type_.primitive), highlevel=False
            ).copy(parameters=type_.parameters)

        return thunk
    else:
        assert len(layout.shape) > 0
        build_tree = recurse(layout._to_regular_primitive(), type_, handle_error)

        def thunk(this):
            return build_tree(this.to_RegularArray())

        return thunk


def recurse_record_any(
    layout: ak.contents.RecordArray,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> BuilderType:
    if isinstance(type_, ak.types.RecordType):
        if len(layout.contents) != len(type_.contents):
            handle_error(
                ValueError(
                    "cannot convert between RecordArray(s) containing different numbers of contents "
                )
            )

        if type_.is_tuple:
            # For tuples, the type order matters. So, we'll permute
            # the layout contents, and keep track of their indices into the `contents` list
            ix_contents = range(len(layout.contents))
            for ix_perm_contents in permutations(ix_contents):
                perm_contents = [layout.contents[i] for i in ix_perm_contents]
                try:
                    builders = [
                        recurse(c, t, raise_invalid_type_conversion_error)
                        for c, t in zip(perm_contents, type_.contents)
                    ]
                except InvalidTypeConversionError:
                    continue
                else:

                    def thunk(this):
                        this_contents = [this.contents[i] for i in ix_perm_contents]
                        return this.copy(
                            fields=None,
                            contents=[b(c) for b, c in zip(builders, this_contents)],
                            parameters=type_.parameters,
                        )

                    return thunk

            handle_error(
                ValueError(
                    "RecordArray(s) can only be converted into RecordArray(s) with compatible contents"
                )
            )

        else:
            type_items = zip(type_.fields, type_.contents)

            # We can permute the type, rather than the layout, because records are not ordered
            for perm in permutations(type_items):
                perm_fields, perm_contents = zip(*perm)
                try:
                    builders = [
                        recurse(c, t, raise_invalid_type_conversion_error)
                        for c, t in zip(layout.contents, perm_contents)
                    ]
                except InvalidTypeConversionError:
                    continue
                else:
                    if not (layout.is_tuple or layout.fields == list(perm_fields)):
                        handle_error(
                            ValueError(
                                "RecordArray(s) can only be converted into RecordArray(s) with compatible "
                                "fields"
                            )
                        )

                    def thunk(this):
                        return this.copy(
                            fields=list(perm_fields),
                            contents=[b(c) for b, c in zip(builders, this.contents)],
                            parameters=type_.parameters,
                        )

                    return thunk
            handle_error(
                ValueError(
                    "RecordArray(s) can only be converted into RecordArray(s) with compatible contents"
                )
            )
    else:
        handle_error(
            ValueError(
                "RecordArray(s) can only be converted into RecordArray(s), options of RecordArray(s), or "
                "unions thereof"
            )
        )


# Require parameters be conserved
def recurse(
    layout: ak.contents.Content,
    type_: ak.types.Type,
    handle_error: ErrorHandlerType,
) -> Callable[[ak.contents.Content], ak.contents.Content]:
    """
    Args:
        layout: layout to recurse into
        type_: expected type of the recursion result
        handle_error: error handler, used instead of bare `raise`

    Returns a callable which converts from `layout` to `type_`. This ensures that
    calling `recurse` is a type-only program, whilst keeping the conversion logic
    adjacent to the type check logic.
    """

    if layout.is_unknown:
        return recurse_unknown_any(layout, type_, handle_error)

    elif layout.is_option:
        return recurse_option_any(layout, type_, handle_error)

    elif layout.is_indexed:
        return recurse_indexed_any(layout, type_, handle_error)

    # Here we *don't* have any layouts that are options, unknowns, or indexed
    # If we see an option, we are therefore *adding* one
    elif isinstance(type_, ak.types.OptionType):
        return recurse_any_option(layout, type_, handle_error)

    elif layout.is_union:
        return recurse_union_any(layout, type_, handle_error)

    # Here we *don't* have any layouts that are options, unknowns, indexed, or unions
    # If we see a union, we are therefore *adding* one
    elif isinstance(type_, ak.types.UnionType):
        return recurse_any_union(layout, type_, handle_error)

    # If we have a list, but it's not supposed to be traversed into
    if layout.is_list and layout.purelist_depth == 1:
        return recurse_list_1d(layout, type_, handle_error)

    elif layout.is_regular or layout.is_list:
        return recurse_list_or_regular_any(layout, type_, handle_error)

    elif layout.is_numpy:
        return recurse_numpy_any(layout, type_, handle_error)

    elif layout.is_record:
        return recurse_record_any(layout, type_, handle_error)
    else:
        handle_error(NotImplementedError(type(layout), type_))


def _impl(array, type_, highlevel, behavior):
    layout = ak.to_layout(array)

    if isinstance(type_, str):
        type_ = ak.types.from_datashape(type_, highlevel=False)

    builder = recurse(layout, type_, raise_error)
    out = builder(layout)
    return wrap_layout(out, like=array, behavior=behavior, highlevel=highlevel)
