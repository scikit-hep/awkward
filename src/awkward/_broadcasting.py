# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy
import enum
import functools
import itertools
from collections.abc import Sequence

import awkward as ak
from awkward._backends.backend import Backend
from awkward._backends.dispatch import backend_of
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._parameters import (
    parameters_are_empty,
    parameters_are_equal,
    parameters_intersect,
)
from awkward._typing import Any, JSONMapping, List, TypedDict
from awkward._util import UNSET, Sentinel
from awkward.contents.bitmaskedarray import BitMaskedArray
from awkward.contents.bytemaskedarray import ByteMaskedArray
from awkward.contents.content import Content
from awkward.contents.emptyarray import EmptyArray
from awkward.contents.indexedarray import IndexedArray
from awkward.contents.indexedoptionarray import IndexedOptionArray
from awkward.contents.listarray import ListArray
from awkward.contents.listoffsetarray import ListOffsetArray
from awkward.contents.numpyarray import NumpyArray
from awkward.contents.recordarray import RecordArray
from awkward.contents.regulararray import RegularArray
from awkward.contents.unionarray import UnionArray
from awkward.contents.unmaskedarray import UnmaskedArray
from awkward.index import (  # IndexU8,  ; Index32,  ; IndexU32,  ; noqa: F401
    Index8,
    Index64,
)
from awkward.record import Record

np = NumpyMetadata.instance()
numpy = Numpy.instance()

optiontypes = (IndexedOptionArray, ByteMaskedArray, BitMaskedArray, UnmaskedArray)
listtypes = (ListOffsetArray, ListArray, RegularArray)


class BroadcastOptions(TypedDict):
    allow_records: bool
    left_broadcast: bool
    right_broadcast: bool
    numpy_to_regular: bool
    regular_to_jagged: bool
    function_name: str | None
    broadcast_parameters_rule: BroadcastParameterRule


def length_of_broadcast(inputs: Sequence) -> int | type[unknown_length]:
    max_length: int | None = None
    has_seen_unknown_length: bool = False
    for x in inputs:
        if not isinstance(x, Content):
            continue
        if x.length is unknown_length:
            has_seen_unknown_length = True
        elif max_length is None:
            max_length = x.length
        else:
            max_length = max(max_length, x.length)

    if has_seen_unknown_length:
        if max_length is None:
            return unknown_length
        else:
            return max_length
    elif max_length is None:
        return 1
    else:
        return max_length


def broadcast_pack(inputs: Sequence, isscalar: list[bool]) -> list:
    maxlen = length_of_broadcast(inputs)
    nextinputs = []

    for x in inputs:
        if isinstance(x, Record):
            index = x.backend.index_nplike.full(maxlen, x.at, dtype=np.int64)
            nextinputs.append(RegularArray(x.array[index], maxlen, 1))
            isscalar.append(True)
        elif isinstance(x, Content):
            nextinputs.append(
                RegularArray(
                    x,
                    x.length,
                    1,
                    parameters=None,
                )
            )
            isscalar.append(False)
        else:
            nextinputs.append(x)
            isscalar.append(True)

    return nextinputs


def broadcast_unpack(x, isscalar: list[bool]):
    if all(isscalar):
        if x.length is not unknown_length and x.length == 0:
            return x._getitem_nothing()._getitem_nothing()
        else:
            return x[0][0]
    else:
        if x.length is not unknown_length and x.length == 0:
            return x._getitem_nothing()
        else:
            return x[0]


def in_function(options):
    if options["function_name"] is None:
        return ""
    else:
        return " in " + options["function_name"]


def ensure_common_length(inputs, options: BroadcastOptions) -> ShapeItem:
    it = iter(inputs)
    length: ShapeItem = unknown_length
    for content in it:
        if content.length is not unknown_length:
            length = content.length
            break

    for other_content in it:
        if other_content.length is unknown_length:
            continue
        if other_content.length != length:
            raise ValueError(
                f"cannot broadcast {type(content).__name__} of length {length} with {type(other_content).__name__} of length {other_content.length}{in_function(options)}"
            )
    return length


def offset_source_rank(list_content: ak.contents.Content) -> int:
    if isinstance(list_content, ak.contents.ListOffsetArray):
        return 0
    elif isinstance(list_content, ak.contents.ListArray):
        return 1
    else:
        raise AssertionError


NO_PARAMETERS = Sentinel("NO_PARAMETERS", __name__)


class BroadcastParameterRule(str, enum.Enum):
    """Behaviour for parameter coalescence during broadcasting."""

    INTERSECT = "intersect"
    ONE_TO_ONE = "one_to_one"
    ALL_OR_NOTHING = "all_or_nothing"
    NONE = "none"


def parameters_of(obj: Any, default: Any = NO_PARAMETERS) -> JSONMapping | None:
    """
    Args:
        obj: #ak.contents.Content that holds parameters, or object
        default: value to return if obj is not an #ak.contents.Content

    Return the parameters of an object if it is a #ak.contents.Content;
    otherwise, return a default value.
    """
    if isinstance(obj, ak.contents.Content):
        return obj._parameters
    else:
        return default


def all_or_nothing_parameters_factory(
    parameters: Sequence[JSONMapping | None], n_outputs: int
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "all or nothing rule":

    If the parameters of all the given contents are equal, then the first content's
    parameters are repeated, i.e. `[parameters, parameters, ...]`. Otherwise, a list
    of Nones is returned, i.e. `[None, None, ...]`.
    """
    input_parameters = [p for p in parameters if p is not NO_PARAMETERS]

    result = None
    if len(input_parameters) > 0:
        # All parameters must match this first layout's parameters
        first_parameters = input_parameters[0]
        # Ensure all parameters match, or set parameters to None
        for other_parameters in input_parameters[1:]:
            if not parameters_are_equal(first_parameters, other_parameters):
                break
        else:
            result = first_parameters

    # NB: we don't make unique copies here, so let's hope everyone
    # is well-behaved downstream!
    return [result] * n_outputs


def intersection_parameters_factory(
    parameters: Sequence[JSONMapping | None],
    n_outputs: int,
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "intersection rule":

    The intersection of `content._parameters.items()` for each content is computed.
    If any parameter dictionaries are None, then a list of Nones is returned, i.e.
    `[None, None, ...]`; otherwise, the computed parameter dictionary is repeated,
    i.e. `[parameters, parameters, ...]`.
    """
    input_parameters = [p for p in parameters if p is not NO_PARAMETERS]

    intersected_parameters = None
    parameters_to_intersect = []
    # Build a list of set-like dict.items() views.
    # If we encounter None-parameters, then we stop early
    # as there can be no intersection.
    for params in input_parameters:
        if parameters_are_empty(params):
            break
        else:
            parameters_to_intersect.append(params)
    # Otherwise, build the intersected parameter dict
    else:
        if len(parameters_to_intersect):
            intersected_parameters = functools.reduce(
                parameters_intersect, parameters_to_intersect
            )
        else:
            intersected_parameters = None

    # NB: we don't make unique copies here, so let's hope everyone
    # is well-behaved downstream!
    return [intersected_parameters] * n_outputs


def one_to_one_parameters_factory(
    parameters: Sequence[JSONMapping | None],
    n_outputs: int,
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using a "one-to-one rule":

    The requested number of outputs is compared against the length of the given
    `inputs`. If the two values match, then a list of parameter objects is returned,
    where each element of the returned list corresponds to the parameters of the
    content at the same position in the `inputs` sequence. If the length of the
    given contents does not match the requested list length, the intersection of the parameters
    is returned instead.
    """
    if n_outputs == len(parameters):
        return [p if p is not NO_PARAMETERS else None for p in parameters]
    else:
        return intersection_parameters_factory(parameters, n_outputs)


def none_parameters_factory(
    parameters: Sequence[JSONMapping | None],
    n_outputs: int,
) -> List[JSONMapping | None]:
    """
    Args:
        parameters: sequence of #ak.contents.Content or other objects
        n_outputs: required number of outputs

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "all or nothing rule":

    A list of Nones is returned, with a length corresponding to the requested number of
    outputs, i.e. `[None, None, ...]`.
    """

    return [None] * n_outputs


def is_string_like(obj) -> bool:
    return (
        isinstance(obj, ak.contents.Content)
        and obj.is_list
        and obj.parameter("__array__") in {"string", "bytestring"}
    )


# Mapping from rule enum values to factory implementations
BROADCAST_RULE_TO_FACTORY_IMPL = {
    BroadcastParameterRule.INTERSECT: intersection_parameters_factory,
    BroadcastParameterRule.ALL_OR_NOTHING: all_or_nothing_parameters_factory,
    BroadcastParameterRule.ONE_TO_ONE: one_to_one_parameters_factory,
    BroadcastParameterRule.NONE: none_parameters_factory,
}


def left_broadcast_to(content: Content, depth: int) -> Content:
    for _ in range(content.purelist_depth, depth):
        content = RegularArray(content, 1, content.length)
    return content


def broadcast_regular_dim_size(contents: Sequence[ak.contents.Content]) -> ShapeItem:
    # Find known size out of our contents
    dim_size: ShapeItem
    it_non_string_regular_contents = iter(
        c for c in contents if c.is_regular and not is_string_like(c)
    )
    for x in it_non_string_regular_contents:
        if x.size is not unknown_length:
            dim_size = x.size
            break
    else:
        # We should only be here if we didn't find any regular arrays with known lengths;
        # there is guaranteed to be at least one non-string list
        dim_size = unknown_length
    # Now we know that we have at least one layout with concrete size, let's check the remainder
    # dim_size=0 should win, though, so we require that dim_size != 0
    if dim_size is not unknown_length and dim_size > 0:
        for x in it_non_string_regular_contents:
            # Any unknown lengths can't be compared
            if x.size is unknown_length:
                continue
            # Any zero-length column triggers zero broadcasting
            if x.size == 0:
                return 0
            else:
                dim_size = max(dim_size, x.size)
    return dim_size


def broadcast_to_offsets_avoiding_carry(
    list_content: ak.contents.Content,
    offsets: ak.index.Index,
) -> ak.contents.Content:
    index_nplike = list_content.backend.index_nplike

    # Without known data, we can't perform these optimisations
    if not index_nplike.known_data:
        return list_content._broadcast_tooffsets64(offsets).content

    elif isinstance(list_content, ListOffsetArray):
        if index_nplike.array_equal(offsets.data, list_content.offsets.data):
            next_length = index_nplike.index_as_shape_item(offsets[-1])
            return list_content.content[:next_length]
        else:
            return list_content._broadcast_tooffsets64(offsets).content
    elif isinstance(list_content, ListArray):
        # Is this list contiguous?
        if index_nplike.array_equal(
            list_content.starts.data[1:], list_content.stops.data[:-1]
        ):
            # Does this list match the offsets?
            if index_nplike.array_equal(
                offsets.data[:-1], list_content.starts.data
            ) and not (
                list_content.stops.data.shape[0] != 0
                and offsets[-1] != list_content.stops[-1]
            ):
                next_length = index_nplike.index_as_shape_item(offsets[-1])
                return list_content.content[:next_length]
            else:
                return list_content._broadcast_tooffsets64(offsets).content
        else:
            return list_content._broadcast_tooffsets64(offsets).content

    elif isinstance(list_content, RegularArray):
        my_offsets = list_content._compact_offsets64(True)
        if index_nplike.array_equal(offsets.data, my_offsets.data):
            return list_content.content[: list_content.size * list_content.length]
        else:
            return list_content._broadcast_tooffsets64(offsets).content

    else:
        raise AssertionError


def apply_step(
    backend: Backend,
    inputs: Sequence,
    action,
    depth: int,
    depth_context,
    lateral_context,
    options: BroadcastOptions,
):
    # This happens when descending anyway, but setting the option does it before action.
    if options["numpy_to_regular"] and any(
        isinstance(x, NumpyArray) and x.data.ndim != 1 for x in inputs
    ):
        inputs = [
            x.to_RegularArray() if isinstance(x, NumpyArray) else x for x in inputs
        ]

    # Rare that any function would want this, but some do.
    if options["regular_to_jagged"] and any(
        isinstance(x, RegularArray) for x in inputs
    ):
        inputs = [
            x.to_ListOffsetArray64(False) if isinstance(x, RegularArray) else x
            for x in inputs
        ]

    contents = [x for x in inputs if isinstance(x, Content)]

    # Handle implicit right-broadcasting (NumPy-like broadcasting).
    if options["right_broadcast"] and any(isinstance(x, listtypes) for x in inputs):
        max_depth = max(x.purelist_depth for x in contents)

        if max_depth > 0 and all(x.purelist_isregular for x in contents):
            nextinputs = [
                left_broadcast_to(o, max_depth) if isinstance(o, Content) else o
                for o in inputs
            ]
            # Did a broadcast take place?
            if any(x is not y for x, y in zip(inputs, nextinputs)):
                return apply_step(
                    backend,
                    nextinputs,
                    action,
                    depth,
                    depth_context,
                    lateral_context,
                    options,
                )

    # Now all lengths must agree
    length = ensure_common_length(contents, options)

    # Load the parameter broadcasting rule implementation
    rule = options["broadcast_parameters_rule"]
    try:
        parameters_factory = BROADCAST_RULE_TO_FACTORY_IMPL[rule]
    except KeyError:
        raise ValueError(
            f"`broadcast_parameters_rule` should be one of {[str(x) for x in BroadcastParameterRule]}, "
            f"but this routine received `{rule}`"
        ) from None

    # This whole function is one big switch statement.
    def broadcast_any_record():
        if not options["allow_records"]:
            raise ValueError(f"cannot broadcast records{in_function(options)}")

        frozen_record_fields: frozenset[str] | None = UNSET
        first_record = next(c for c in contents if c.is_record)
        nextparameters = []

        for x in contents:
            if x.is_record:
                nextparameters.append(x._parameters)

                # Ensure all records are tuples, or all records are records
                if x.is_tuple != first_record.is_tuple:
                    raise TypeError(
                        f"cannot broadcast a tuple against a record{in_function(options)}"
                    )

                # Check fields match for records
                if not x.is_tuple:
                    if frozen_record_fields is UNSET:
                        frozen_record_fields = frozenset(x.fields)
                    elif frozen_record_fields != frozenset(x.fields):
                        raise ValueError(
                            "cannot broadcast records because fields don't "
                            "match{}:\n    {}\n    {}".format(
                                in_function(options),
                                ", ".join(sorted(first_record.fields)),
                                ", ".join(sorted(x.fields)),
                            )
                        )
            else:
                nextparameters.append(NO_PARAMETERS)

        numoutputs = None
        outcontents = []
        for field in first_record.fields:
            outcontents.append(
                apply_step(
                    backend,
                    [
                        x.content(field) if isinstance(x, RecordArray) else x
                        for x in inputs
                    ],
                    action,
                    depth,
                    copy.copy(depth_context),
                    lateral_context,
                    options,
                )
            )
            assert isinstance(outcontents[-1], tuple)
            if numoutputs is None:
                numoutputs = len(outcontents[-1])
            else:
                assert numoutputs == len(outcontents[-1])

        parameters = parameters_factory(nextparameters, numoutputs)

        return tuple(
            RecordArray(
                [x[i] for x in outcontents],
                # Explicitly set fields to None if this is a tuple
                None if first_record.is_tuple else first_record.fields,
                length,
                parameters=p,
            )
            for i, p in enumerate(parameters)
        )

    def broadcast_any_list():
        index_nplike = backend.index_nplike
        # Under the category of "is_list", we have both strings and non-strings
        # The strings should behave like non-lists within these routines.

        # Are the non-string list types exclusively regular?
        if all(x.is_regular or (is_string_like(x) or not x.is_list) for x in contents):
            # Compute the expected dim size
            dim_size = broadcast_regular_dim_size(contents)
            dimsize_maybe_broadcastable = dim_size is unknown_length or dim_size > 1
            dimsize_is_zero = dim_size is not unknown_length and dim_size == 0

            # Build a broadcast index for size=1 contents, and identify whether we have strings
            inputs_are_strings = []
            size_one_carry_index = None
            for x in inputs:
                if isinstance(x, ak.contents.Content):
                    content_is_string = is_string_like(x)
                    inputs_are_strings.append(content_is_string)
                    if (
                        # Strings don't count as lists in this context
                        not content_is_string
                        # Is this layout known to be size==1?
                        and x.is_regular
                        and x.size is not unknown_length
                        and x.size == 1
                        # Does the computed dim_size support broadcasting
                        and dimsize_maybe_broadcastable
                        and size_one_carry_index is None
                    ):
                        # For any (N, 1) array, we know we'll broadcast to (N, M) where M is maxsize
                        size_one_carry_index = Index64(
                            index_nplike.repeat(
                                index_nplike.arange(
                                    index_nplike.shape_item_as_index(length),
                                    dtype=np.int64,
                                ),
                                index_nplike.shape_item_as_index(dim_size),
                            ),
                            nplike=index_nplike,
                        )
                else:
                    inputs_are_strings.append(False)

            # W.r.t broadcasting against other lists, we have three possibilities
            # a. any (exactly) size-0 content broadcasts all other regular dimensions to 0
            # b. any (exactly) size-1 content broadcasts to the `size_one_carry_index``
            # c. otherwise, the list size should equal the dimension; recurse into the content as-is

            # If we have non-lists, these are just appended as-is. As we're dealing with regular layouts,
            # we don't left-broadcast
            nextinputs = []
            nextparameters = []
            for x, x_is_string in zip(inputs, inputs_are_strings):
                if isinstance(x, RegularArray) and not x_is_string:
                    content_size_maybe_one = (
                        x.size is not unknown_length and x.size == 1
                    )
                    # If dimsize is known to be exactly zero, all contents are zero length
                    if dimsize_is_zero:
                        nextinputs.append(x.content[:0])
                        nextparameters.append(x._parameters)
                    # If we have a known size=1 content, then broadcast it to the dimension size
                    elif dimsize_maybe_broadcastable and content_size_maybe_one:
                        nextinputs.append(
                            x.content[: x.length * x.size]._carry(
                                size_one_carry_index, allow_lazy=False
                            )
                        )
                        nextparameters.append(x._parameters)
                    # Any unknown values or sizes are assumed to be correct as-is
                    elif (
                        dim_size is unknown_length
                        or x.size is unknown_length
                        or x.size == dim_size
                    ):
                        nextinputs.append(x.content[: x.length * x.size])
                        nextparameters.append(x._parameters)
                    else:
                        raise ValueError(
                            "cannot broadcast RegularArray of size "
                            f"{x.size} with RegularArray of size {dim_size}{in_function(options)}"
                        )
                else:
                    nextinputs.append(x)
                    nextparameters.append(NO_PARAMETERS)

            outcontent = apply_step(
                backend,
                nextinputs,
                action,
                depth + 1,
                copy.copy(depth_context),
                lateral_context,
                options,
            )
            assert isinstance(outcontent, tuple)
            parameters = parameters_factory(nextparameters, len(outcontent))

            return tuple(
                RegularArray(x, dim_size, length, parameters=p)
                for x, p in zip(outcontent, parameters)
            )
        # General list-handling case: the offsets of each list may be different.
        else:
            offsets_content = None
            input_is_string = []

            # Find the offsets to broadcast to, taking the "best" list
            for x in inputs:
                if isinstance(x, Content):
                    content_is_string = is_string_like(x)
                    input_is_string.append(content_is_string)
                    if (
                        x.is_list
                        and not x.is_regular
                        and not content_is_string
                        and (
                            offsets_content is None
                            or (
                                offset_source_rank(x)
                                < offset_source_rank(offsets_content)
                            )
                        )
                    ):
                        offsets_content = x
                else:
                    input_is_string.append(False)

            # Build the offsets of the lowest-ranking source
            offsets = offsets_content._compact_offsets64(True)

            nextinputs = []
            nextparameters = []
            for x, x_is_string in zip(inputs, input_is_string):
                if isinstance(x, listtypes) and not x_is_string:
                    next_content = broadcast_to_offsets_avoiding_carry(x, offsets)
                    nextinputs.append(next_content)
                    nextparameters.append(x._parameters)
                # Handle implicit left-broadcasting (non-NumPy-like broadcasting).
                elif options["left_broadcast"] and isinstance(x, Content):
                    nextinputs.append(
                        RegularArray(x, 1, x.length)
                        ._broadcast_tooffsets64(offsets)
                        .content
                    )
                    nextparameters.append(NO_PARAMETERS)
                else:
                    nextinputs.append(x)
                    nextparameters.append(NO_PARAMETERS)

            outcontent = apply_step(
                backend,
                nextinputs,
                action,
                depth + 1,
                copy.copy(depth_context),
                lateral_context,
                options,
            )
            assert isinstance(outcontent, tuple)
            parameters = parameters_factory(nextparameters, len(outcontent))

            return tuple(
                ListOffsetArray(offsets, x, parameters=p)
                for x, p in zip(outcontent, parameters)
            )

    def broadcast_any_option():
        mask = None
        for x in contents:
            if x.is_option:
                m = x.mask_as_bool(valid_when=False)
                if mask is None:
                    mask = m
                else:
                    mask = backend.index_nplike.logical_or(mask, m, maybe_out=mask)

        nextmask = Index8(mask.view(np.int8))
        index = backend.index_nplike.full(mask.shape[0], -1, dtype=np.int64)
        index[~mask] = backend.index_nplike.arange(
            backend.index_nplike.shape_item_as_index(mask.shape[0])
            - backend.index_nplike.count_nonzero(mask),
            dtype=np.int64,
        )
        index = Index64(index)
        if any(not x.is_option for x in contents):
            nextindex = backend.index_nplike.arange(
                backend.index_nplike.shape_item_as_index(mask.shape[0]),
                dtype=np.int64,
            )
            nextindex[mask] = -1
            nextindex = Index64(nextindex)

        nextinputs = []
        nextparameters = []
        for x in inputs:
            if isinstance(x, optiontypes):
                nextinputs.append(x.project(nextmask))
                nextparameters.append(x._parameters)
            elif isinstance(x, Content):
                nextinputs.append(IndexedOptionArray(nextindex, x).project(nextmask))
                nextparameters.append(x._parameters)
            else:
                nextinputs.append(x)
                nextparameters.append(NO_PARAMETERS)

        outcontent = apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )
        assert isinstance(outcontent, tuple)
        parameters = parameters_factory(nextparameters, len(outcontent))

        return tuple(
            IndexedOptionArray.simplified(index, x, parameters=p)
            for x, p in zip(outcontent, parameters)
        )

    def broadcast_any_union():
        nextparameters = []

        for x in inputs:
            if isinstance(x, UnionArray):
                nextparameters.append(x._parameters)
            else:
                nextparameters.append(NO_PARAMETERS)

        union_tags, union_num_contents, length = [], [], unknown_length
        for x in contents:
            if x.is_union:
                tags = x.tags.raw(backend.index_nplike)
                union_tags.append(tags)
                union_num_contents.append(len(x.contents))

                if length is unknown_length:
                    length = tags.shape[0]
                elif tags.shape[0] is unknown_length:
                    continue
                elif length != tags.shape[0]:
                    raise ValueError(
                        f"cannot broadcast UnionArray of length {length} "
                        f"with UnionArray of length {tags.shape[0]}{in_function(options)}"
                    )

        tags = backend.index_nplike.empty(length, dtype=np.int8)
        index = backend.index_nplike.empty(length, dtype=np.int64)

        # Stack all union tags
        combos = backend.index_nplike.stack(union_tags, axis=-1)

        # Build array of indices (c1, c2, c3, ..., cn) of contents in
        # (union 1, union 2, union 3, ..., union n)
        all_combos = list(itertools.product(*[range(x) for x in union_num_contents]))

        numoutputs = None
        outcontents = []

        for tag, j_contents in enumerate(all_combos):
            combo = backend.index_nplike.asarray(j_contents, dtype=np.int64)
            mask = backend.index_nplike.all(combos == combo, axis=-1)
            tags[mask] = tag
            index[mask] = backend.index_nplike.arange(
                backend.index_nplike.count_nonzero(mask), dtype=np.int64
            )
            nextinputs = []
            it_j_contents = iter(j_contents)
            for x in inputs:
                if isinstance(x, UnionArray):
                    nextinputs.append(x[mask].project(next(it_j_contents)))
                elif isinstance(x, Content):
                    nextinputs.append(x[mask])
                else:
                    nextinputs.append(x)
            outcontents.append(
                apply_step(
                    backend,
                    nextinputs,
                    action,
                    depth,
                    copy.copy(depth_context),
                    lateral_context,
                    options,
                )
            )
            assert isinstance(outcontents[-1], tuple)
            if numoutputs is None:
                numoutputs = len(outcontents[-1])
            else:
                assert numoutputs == len(outcontents[-1])

        assert numoutputs is not None

        parameters = parameters_factory(nextparameters, numoutputs)

        return tuple(
            UnionArray.simplified(
                Index8(tags),
                Index64(index),
                [x[i] for x in outcontents],
                parameters=p,
            )
            for i, p in enumerate(parameters)
        )

    def broadcast_any_indexed():
        # The `apply` function may exit at the level of a `RecordArray`. We can avoid projection
        # of the record array in such cases, in favour of a deferred carry. This can be done by
        # "pushing" the `IndexedArray` _into_ the record (i.e., wrapping each `content`).
        nextinputs = [
            x._push_inside_record_or_project() if isinstance(x, IndexedArray) else x
            for x in inputs
        ]
        return apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

    def broadcast_any_nd_numpy():
        nextinputs = [
            x.to_RegularArray() if isinstance(x, NumpyArray) else x for x in inputs
        ]
        return apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

    def broadcast_any_unknown():
        nextinputs = [
            x.to_NumpyArray(np.float64, backend) if isinstance(x, EmptyArray) else x
            for x in inputs
        ]
        return apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            options,
        )

    def continuation():
        # Any EmptyArrays?
        if any(x.is_unknown for x in contents):
            return broadcast_any_unknown()

        # Any NumpyArrays with ndim != 1?
        elif any(x.is_numpy and x.purelist_depth != 1 for x in contents):
            return broadcast_any_nd_numpy()

        # Any IndexedArrays?
        elif any((x.is_indexed and not x.is_option) for x in contents):
            return broadcast_any_indexed()

        # Any UnionArrays?
        elif any(x.is_union for x in contents):
            return broadcast_any_union()

        # Any option-types?
        elif any(x.is_option for x in contents):
            return broadcast_any_option()

        # Any non-string list-types?
        elif any(x.is_list and not is_string_like(x) for x in contents):
            return broadcast_any_list()

        # Any RecordArrays?
        elif any(x.is_record for x in contents):
            return broadcast_any_record()

        else:
            raise ValueError(
                "cannot broadcast: {}{}".format(
                    ", ".join(repr(type(x)) for x in inputs), in_function(options)
                )
            )

    result = action(
        inputs,
        depth=depth,
        depth_context=depth_context,
        lateral_context=lateral_context,
        continuation=continuation,
        backend=backend,
        options=options,
    )

    if isinstance(result, tuple) and all(isinstance(x, Content) for x in result):
        if any(content.backend is not backend for content in result):
            raise ValueError(
                "broadcasting action returned layouts with different backends: ",
                ", ".join([content.backend.name for content in result]),
            )
        return result
    elif result is None:
        return continuation()
    else:
        raise AssertionError(result)


def broadcast_and_apply(
    inputs,
    action,
    *,
    depth_context: dict[str, Any] | None = None,
    lateral_context: dict[str, Any] | None = None,
    allow_records: bool = True,
    left_broadcast: bool = True,
    right_broadcast: bool = True,
    numpy_to_regular: bool = False,
    regular_to_jagged: bool = False,
    function_name: str | None = None,
    broadcast_parameters_rule=BroadcastParameterRule.INTERSECT,
):
    # Expect arrays to already have common backend
    backend = backend_of(*inputs, coerce_to_common=False)
    isscalar = []
    out = apply_step(
        backend,
        broadcast_pack(inputs, isscalar),
        action,
        0,
        depth_context,
        lateral_context,
        {
            "allow_records": allow_records,
            "left_broadcast": left_broadcast,
            "right_broadcast": right_broadcast,
            "numpy_to_regular": numpy_to_regular,
            "regular_to_jagged": regular_to_jagged,
            "function_name": function_name,
            "broadcast_parameters_rule": broadcast_parameters_rule,
        },
    )
    assert isinstance(out, tuple)
    return tuple(broadcast_unpack(x, isscalar) for x in out)
