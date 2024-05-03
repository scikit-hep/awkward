# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from itertools import permutations

import awkward as ak
from awkward._backends.dispatch import backend_of_obj
from awkward._dispatch import high_level_function
from awkward._do import mergeable
from awkward._layout import HighLevelContext, ensure_same_backend, maybe_posaxis
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._parameters import type_parameters_equal
from awkward._regularize import regularize_axis
from awkward._typing import Sequence
from awkward.contents import Content
from awkward.operations.ak_fill_none import fill_none
from awkward.types.numpytype import primitive_to_dtype

__all__ = ("concatenate",)

np = NumpyMetadata.instance()


@ak._connect.numpy.implements("concatenate")
@high_level_function()
def concatenate(
    arrays, axis=0, *, mergebool=True, highlevel=True, behavior=None, attrs=None
):
    """
    Args:
        arrays: Array-like data (anything #ak.to_layout recognizes).
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        mergebool (bool): If True, boolean and numeric data can be combined
            into the same buffer, losing information about False vs `0` and
            True vs `1`; otherwise, they are kept in separate buffers with
            distinct types (using an #ak.contents.UnionArray).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns an array with `arrays` concatenated. For `axis=0`, this means that
    one whole array follows another. For `axis=1`, it means that the `arrays`
    must have the same lengths and nested lists are each concatenated,
    element for element, and similarly for deeper levels.
    """
    # Dispatch
    if (
        # Is an array with a known backend
        backend_of_obj(arrays, default=None) is not None
    ):
        yield (arrays,)
    else:
        yield arrays

    # Implementation
    return _impl(arrays, axis, mergebool, highlevel, behavior, attrs)


def _merge_as_union(
    contents: Sequence[Content], parameters=None
) -> ak.contents.UnionArray:
    length = sum(c.length for c in contents)
    first = contents[0]
    tags = ak.index.Index8.empty(length, first.backend.index_nplike)
    index = ak.index.Index64.empty(length, first.backend.index_nplike)

    offset = 0
    for i, content in enumerate(contents):
        content.backend.maybe_kernel_error(
            content.backend["awkward_UnionArray_filltags_const", tags.dtype.type](
                tags.data, offset, content.length, i
            )
        )
        content.backend.maybe_kernel_error(
            content.backend["awkward_UnionArray_fillindex_count", index.dtype.type](
                index.data, offset, content.length
            )
        )
        offset += content.length

    return ak.contents.UnionArray.simplified(
        tags, index, contents, parameters=parameters
    )


def _impl(arrays, axis, mergebool, highlevel, behavior, attrs):
    axis = regularize_axis(axis)
    # Simple single-array, axis=0 fast-path
    if (
        # Is an array with a known backend
        backend_of_obj(arrays, default=None) is not None
    ):
        # Convert the array to a layout object
        content = ak.operations.to_layout(
            arrays, allow_record=False, primitive_policy="error"
        )
        # Only handle concatenation along `axis=0`
        # Let ambiguous depth arrays fall through
        if maybe_posaxis(content, axis, 1) == 0:
            return ak.operations.ak_flatten._impl(arrays, 1, highlevel, behavior, attrs)

    # Now that we're sure `arrays` is not a singular array
    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        content_or_others = ensure_same_backend(
            *(
                ctx.unwrap(
                    x,
                    allow_record=axis != 0,
                    allow_unknown=False,
                    primitive_policy="pass-through",
                )
                for x in arrays
            )
        )

    contents = [x for x in content_or_others if isinstance(x, ak.contents.Content)]
    if len(contents) == 0:
        raise ValueError("need at least one array to concatenate")

    posaxis = maybe_posaxis(contents[0], axis, 1)
    maxdepth = max(
        x.minmax_depth[1]
        for x in content_or_others
        if isinstance(x, ak.contents.Content)
    )
    if posaxis is None or not 0 <= posaxis < maxdepth:
        raise ValueError(
            f"axis={axis} is beyond the depth of this array or the depth of this array "
            "is ambiguous"
        )
    for x in content_or_others:
        if isinstance(x, ak.contents.Content):
            if maybe_posaxis(x, axis, 1) != posaxis:
                raise ValueError(
                    "arrays to concatenate do not have the same depth for negative "
                    f"axis={axis}"
                )

    if posaxis == 0:
        content_or_others = [
            x if isinstance(x, ak.contents.Content) else ak.operations.to_layout([x])
            for x in content_or_others
        ]
        batches = [[content_or_others[0]]]
        for x in content_or_others[1:]:
            batch = batches[-1]
            if ak._do.mergeable(batch[-1], x, mergebool=mergebool):
                batch.append(x)
            else:
                batches.append([x])

        contents = [ak._do.mergemany(b) for b in batches]
        if len(contents) > 1:
            out = _merge_as_union(contents)
        else:
            out = contents[0]

        if isinstance(out, ak.contents.UnionArray):
            out = type(out).simplified(
                out._tags,
                out._index,
                out._contents,
                parameters=out._parameters,
                mergebool=mergebool,
            )

    else:

        def action(inputs, depth, backend, **kwargs):
            if any(
                x.minmax_depth == (1, 1)
                for x in inputs
                if isinstance(x, ak.contents.Content)
            ):
                raise ValueError(
                    "at least one array is not deep enough to concatenate at "
                    f"axis={axis}"
                )

            if depth != posaxis:
                return

            if any(isinstance(x, ak.contents.Content) and x.is_option for x in inputs):
                nextinputs = []
                for x in inputs:
                    if x.is_option and x.content.is_list:
                        empty = ak.to_backend([], backend)
                        nextinputs.append(fill_none(x, empty, axis=0, highlevel=False))
                    else:
                        nextinputs.append(x)
                inputs = nextinputs

            # Ensure the lengths agree, taking known lengths over unknown lengths
            length = None
            for x in inputs:
                if isinstance(x, ak.contents.Content):
                    if length is None:
                        length = x.length
                    elif x.length is unknown_length:
                        continue
                    elif length is unknown_length:
                        length = x.length
                    elif length != x.length:
                        raise ValueError(
                            f"all arrays must have the same length for axis={axis}"
                        )
            assert length is not None

            if all(
                (isinstance(x, ak.contents.Content) and x.is_regular)
                or (isinstance(x, ak.contents.NumpyArray) and x.data.ndim > 1)
                or not isinstance(x, ak.contents.Content)
                for x in inputs
            ):
                regulararrays = []
                sizes = []
                for x in inputs:
                    if isinstance(x, ak.contents.RegularArray):
                        regulararrays.append(x)
                    elif isinstance(x, ak.contents.NumpyArray):
                        regulararrays.append(x.to_RegularArray())
                    else:
                        regulararrays.append(
                            ak.contents.RegularArray(
                                ak.contents.NumpyArray(
                                    backend.nplike.broadcast_to(
                                        backend.nplike.asarray([x]), (length,)
                                    )
                                ),
                                1,
                            )
                        )
                    sizes.append(regulararrays[-1].size)

                prototype = backend.index_nplike.empty(sum(sizes), dtype=np.int8)
                start = 0
                for tag, size in enumerate(sizes):
                    prototype[start : start + size] = tag
                    start += size

                tags = ak.index.Index8(
                    backend.index_nplike.reshape(
                        backend.index_nplike.broadcast_to(
                            prototype, (length, prototype.size)
                        ),
                        (-1,),
                    )
                )
                index = ak.contents.UnionArray.regular_index(tags, backend=backend)
                inner = ak.contents.UnionArray.simplified(
                    tags,
                    index,
                    [x._content for x in regulararrays],
                    mergebool=mergebool,
                )

                return (ak.contents.RegularArray(inner, prototype.size),)

            elif all(
                isinstance(x, ak.contents.Content)
                and x.is_list
                or (isinstance(x, ak.contents.NumpyArray) and x.data.ndim > 1)
                or not isinstance(x, ak.contents.Content)
                for x in inputs
            ):
                nextinputs = []
                for x in inputs:
                    if isinstance(x, ak.contents.Content):
                        nextinputs.append(x)
                    else:
                        nextinputs.append(
                            ak.contents.ListOffsetArray(
                                ak.index.Index64(
                                    backend.index_nplike.arange(
                                        backend.index_nplike.shape_item_as_index(
                                            length + 1
                                        ),
                                        dtype=np.int64,
                                    ),
                                    nplike=backend.index_nplike,
                                ),
                                ak.contents.NumpyArray(
                                    backend.nplike.broadcast_to(
                                        backend.nplike.asarray([x]), (length,)
                                    )
                                ),
                            )
                        )

                counts = backend.index_nplike.zeros(
                    nextinputs[0].length, dtype=np.int64
                )
                all_counts = []
                all_flatten = []

                for x in nextinputs:
                    o, f = x._offsets_and_flattened(1, 1)
                    c = o.data[1:] - o.data[:-1]
                    backend.index_nplike.add(counts, c, maybe_out=counts)
                    all_counts.append(c)
                    all_flatten.append(f)

                offsets = backend.index_nplike.empty(
                    nextinputs[0].length + 1, dtype=np.int64
                )
                offsets[0] = 0
                backend.index_nplike.cumsum(counts, maybe_out=offsets[1:])

                offsets = ak.index.Index64(offsets, nplike=backend.index_nplike)

                tags, index = ak.contents.UnionArray.nested_tags_index(
                    offsets,
                    [ak.index.Index64(x) for x in all_counts],
                    backend=backend,
                )

                inner = ak.contents.UnionArray.simplified(
                    tags, index, all_flatten, mergebool=mergebool
                )

                return (ak.contents.ListOffsetArray(offsets, inner),)

            else:
                return None

        out = ak._broadcasting.broadcast_and_apply(
            content_or_others, action, allow_records=True, right_broadcast=False
        )[0]

    return ctx.wrap(out, highlevel=highlevel)


def _form_has_type(form, type_):
    """
    Args:
        form: content object
        type_: low-level type object

    Returns True if the form satisfies the given type; otherwise False.
    """
    if not type_parameters_equal(form._parameters, type_._parameters):
        return False

    if form.is_unknown:
        return isinstance(type_, ak.types.UnknownType)
    elif form.is_option:
        return isinstance(type_, ak.types.OptionType) and _form_has_type(
            form.content, type_.content
        )
    elif form.is_indexed:
        return _form_has_type(form.content, type_)
    elif form.is_regular:
        return (
            isinstance(type_, ak.types.RegularType)
            and (
                form.size is unknown_length
                or type_.size is unknown_length
                or form.size == type_.size
            )
            and _form_has_type(form.content, type_.content)
        )
    elif form.is_list:
        return isinstance(type_, ak.types.ListType) and _form_has_type(
            form.content, type_.content
        )
    elif form.is_numpy:
        for _ in range(form.purelist_depth - 1):
            if not isinstance(type_, ak.types.RegularType):
                return False
            type_ = type_.content
        return (
            isinstance(type_, ak.types.NumpyType) and form.primitive == type_.primitive
        )
    elif form.is_record:
        if (
            not isinstance(type_, ak.types.RecordType)
            or type_.is_tuple != form.is_tuple
        ):
            return False

        if form.is_tuple:
            return all(
                _form_has_type(c, t) for c, t in zip(form.contents, type_.contents)
            )
        else:
            return (frozenset(form.fields) == frozenset(type_.fields)) and all(
                _form_has_type(form.content(f), type_.content(f)) for f in type_.fields
            )
    elif form.is_union:
        if len(form.contents) != len(type_.contents):
            return False

        for contents in permutations(form.contents):
            if all(
                _form_has_type(form, type_)
                for form, type_ in zip(contents, type_.contents)
            ):
                return True
        return False
    else:
        raise TypeError(form)


# This routine should not try to replicate the merge logic,
# but we can make use of assumptions w.r.t to what the merge will do.
# e.g., merging can add new unions, promote to options, change dtypes of NumPy arrays
def enforce_concatenated_form(layout, form):
    # Merge invariant (drop known-ness)
    if not layout.is_unknown and form.is_unknown:
        raise AssertionError(
            "merge result should never be of an unknown type unless the layout is unknown"
        )
    # Unknowns become canonical forms
    elif layout.is_unknown and not form.is_unknown:
        return form.length_zero_array().to_backend(layout.backend)

    ############## Unions #####################################################
    # Merge invariant (drop union)
    elif layout.is_union and not form.is_union:
        raise AssertionError("merge result should be a union if layout is a union")
    # Add a union
    elif not layout.is_union and form.is_union:
        # Merge invariant (unions are i8-i64)
        if not (form.tags == "i8" and form.index == "i64"):
            raise AssertionError(
                "merge result that forms a union should have i8 tags and i64 index"
            )

        # Non-categoricals can be merged into union
        if (
            layout.is_indexed
            and not layout.is_option
            and layout.parameter("__array__") != "categorical"
        ):
            index = layout.index.to64()
            # Take the content and drop the parameters (we're taking parameters from form!)
            layout_to_merge = layout.content
        # Otherwise, we move into the contents
        else:
            index = ak.index.Index64(
                layout.backend.index_nplike.arange(layout.length, dtype=np.int64)
            )
            layout_to_merge = layout

        type_ = layout_to_merge.form.type

        # First assume this type is exactly represented in the union.
        # This won't hold true if any (and not all) of the contents are an option
        # Or if there were mergeable (but non-equal type) pairs in the original
        # concatenation that formed this union
        union_has_exact_type = False
        contents = []
        for content_form in form.contents:
            if _form_has_type(content_form, type_):
                contents.insert(
                    0, enforce_concatenated_form(layout_to_merge, content_form)
                )
                union_has_exact_type = True
            else:
                contents.append(
                    content_form.length_zero_array().to_backend(layout.backend)
                )

        # Otherwise, find anything we can merge with
        if not union_has_exact_type:
            contents.clear()

            for content_form in form.contents:
                # TODO check forms mergeable
                content_layout = content_form.length_zero_array().to_backend(
                    layout.backend
                )
                if mergeable(content_layout, layout_to_merge):
                    contents.insert(
                        0, enforce_concatenated_form(layout_to_merge, content_form)
                    )
                else:
                    contents.append(
                        content_form.length_zero_array().to_backend(layout.backend)
                    )

        return ak.contents.UnionArray(
            ak.index.Index8(
                layout.backend.index_nplike.zeros(layout.length, dtype=np.int8)
            ),
            index,
            contents,
            parameters=form._parameters,
        )
    # Preserve union
    elif layout.is_union and form.is_union:
        # Merge invariant (unions are i8-i64)
        if not (form.tags == "i8" and form.index == "i64"):
            raise AssertionError(
                "merge result that forms a union should have i8 tags and i64 index"
            )
        if len(form.contents) < len(layout.contents):
            raise AssertionError(
                "merge result should only grow or preserve a union's cardinality"
            )
        form_contents = [
            f.length_zero_array().to_backend(layout.backend) for f in form.contents
        ]
        form_indices = range(len(form_contents))
        for form_projection_indices in permutations(form_indices, len(layout.contents)):
            if all(
                mergeable(c, form_contents[i])
                for c, i in zip(layout.contents, form_projection_indices)
            ):
                break
        else:
            raise AssertionError(
                "merge result should be mergeable against some permutation of the layout"
            )

        next_contents = [
            enforce_concatenated_form(c, form.contents[i])
            for c, i in zip(layout.contents, form_projection_indices)
        ]
        next_contents.extend(
            [
                form_contents[i]
                for i in (set(form_indices) - set(form_projection_indices))
            ]
        )
        return ak.contents.UnionArray(
            ak.index.Index8(
                layout.backend.index_nplike.astype(layout.tags.data, np.int8)
            ),
            layout.index.to64(),
            next_contents,
            parameters=form._parameters,
        )

    ############## Options ####################################################
    # Merge invariant (drop option)
    elif layout.is_option and not form.is_option:
        raise AssertionError("merge result should be an option if layout is an option")
    # Add option
    elif not layout.is_option and form.is_option:
        return enforce_concatenated_form(
            ak.contents.UnmaskedArray.simplified(layout), form
        )
    # Preserve option
    elif layout.is_option and form.is_option:
        if isinstance(form, ak.forms.IndexedOptionForm):
            if form.index != "i64":
                raise AssertionError(
                    "IndexedOptionForm should have i64 for merge results"
                )
            return layout.to_IndexedOptionArray64().copy(
                content=enforce_concatenated_form(layout.content, form.content),
                parameters=form._parameters,
            )
        # Non IndexedOptionArray types require all merge candidates to have same form
        elif isinstance(
            form,
            (ak.forms.ByteMaskedForm, ak.forms.BitMaskedForm, ak.forms.UnmaskedForm),
        ):
            return layout.copy(
                content=enforce_concatenated_form(layout.content, form.content),
                parameters=form._parameters,
            )
        else:
            raise AssertionError

    ############## Indexed ####################################################
    # Merge invariant (drop indexed)
    elif layout.is_indexed and not form.is_indexed:
        raise AssertionError("merge result must be indexed if layout is indexed")
    # Add index
    elif not layout.is_indexed and form.is_indexed:
        return ak.contents.IndexedArray(
            ak.index.Index64(layout.backend.index_nplike.arange(layout.length)),
            enforce_concatenated_form(layout, form.content),
            parameters=form._parameters,
        )
    # Preserve index
    elif layout.is_indexed and form.is_indexed:
        if form.index != "i64":
            raise AssertionError("merge result must be i64")
        return ak.contents.IndexedArray(
            layout.index.to64(),
            content=enforce_concatenated_form(layout.content, form.content),
            parameters=form._parameters,
        )

    ############## NumPy ######################################################
    elif layout.is_numpy and form.is_numpy:
        if layout.inner_shape != form.inner_shape:
            raise AssertionError("layout must have same inner_shape as merge result")

        return ak.values_astype(
            # HACK: drop parameters from type so that character arrays are supported
            layout.copy(parameters=None),
            to=primitive_to_dtype(form.primitive),
            highlevel=False,
        ).copy(parameters=form._parameters)

    ############## Lists ######################################################
    # Merge invariant (regular to numpy)
    elif layout.is_regular and form.is_numpy:
        raise AssertionError("layout cannot be regular for NumpyForm merge result")
    # Merge invariant (ragged to regular)
    elif not (layout.is_regular or layout.is_numpy) and form.is_regular:
        raise AssertionError("merge result should be ragged if any input is ragged")
    elif layout.is_numpy and form.is_list:
        if len(layout.inner_shape) == 0:
            raise AssertionError("layout must be at least 2D if merge result is a list")
        return enforce_concatenated_form(layout.to_RegularArray(), form)
    elif layout.is_regular and form.is_regular:
        # regular → regular requires same size!
        if layout.size != form.size:
            raise AssertionError(
                "RegularForm must have same size as layout for merge result"
            )
        return layout.copy(
            content=enforce_concatenated_form(layout.content, form.content),
            parameters=form._parameters,
        )
    elif layout.is_regular and form.is_list:
        if isinstance(form, (ak.forms.ListOffsetForm, ak.forms.ListForm)):
            return enforce_concatenated_form(layout.to_ListOffsetArray64(False), form)
        else:
            raise AssertionError
    elif layout.is_list and form.is_list:
        if isinstance(form, ak.forms.ListOffsetForm):
            layout = layout.to_ListOffsetArray64(False)
            return layout.copy(
                content=enforce_concatenated_form(layout.content, form.content),
                parameters=form._parameters,
            )
        elif isinstance(form, ak.forms.ListForm):
            if not (form.starts == "i64" and form.stops == "i64"):
                raise TypeError("ListForm should have i64 for merge results")
            return ak.contents.ListArray(
                layout.starts.to64(),
                layout.stops.to64(),
                enforce_concatenated_form(layout.content, form.content),
                parameters=form._parameters,
            )
        else:
            raise AssertionError

    ############## Records ####################################################
    # Merge invariant (mix record-tuple)
    elif layout.is_record and not form.is_record:
        raise AssertionError("merge result should be a record if layout is a record")
    # Merge invariant (mix record-tuple)
    elif not layout.is_record and form.is_record:
        raise AssertionError(
            "layout result should be a record if merge result is a record"
        )
    elif layout.is_record and form.is_record:
        if frozenset(layout.fields) != frozenset(form.fields):
            raise AssertionError("merge result and form must have matching fields")
        elif layout.is_tuple != form.is_tuple:
            raise AssertionError(
                "merge result and form must both be tuple or record-like"
            )
        return ak.contents.RecordArray(
            [
                enforce_concatenated_form(layout.content(f), form.content(f))
                for f in layout.fields
            ],
            layout._fields,
            length=layout.length,
            parameters=form._parameters,
            backend=layout.backend,
        )
    else:
        raise NotImplementedError
