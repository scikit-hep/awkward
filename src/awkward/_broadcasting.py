# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
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
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._parameters import (
    parameters_are_empty,
    parameters_are_equal,
    parameters_intersect,
)
from awkward._typing import Any, Callable, Dict, List, TypeAlias, Union
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
    Index,
    Index8,
    Index64,
)
from awkward.record import Record

np = NumpyMetadata.instance()
numpy = Numpy.instance()

optiontypes = (IndexedOptionArray, ByteMaskedArray, BitMaskedArray, UnmaskedArray)
listtypes = (ListOffsetArray, ListArray, RegularArray)


def length_of_broadcast(inputs: Sequence) -> int | type[unknown_length]:
    maxlen = -1

    for x in inputs:
        if isinstance(x, Content):
            if x.length is unknown_length:
                return x.length

            maxlen = max(maxlen, x.length)

    if maxlen < 0:
        maxlen = 1

    return maxlen


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


def broadcast_unpack(x, isscalar: list[bool], backend: Backend):
    if all(isscalar):
        if not backend.nplike.known_data or x.length == 0:
            return x._getitem_nothing()._getitem_nothing()
        else:
            return x[0][0]
    else:
        if not backend.nplike.known_data or x.length == 0:
            return x._getitem_nothing()
        else:
            return x[0]


def in_function(options):
    if options["function_name"] is None:
        return ""
    else:
        return " in " + options["function_name"]


def checklength(inputs, options):
    it = iter(inputs)
    length: int
    for content in it:
        if content.length is not unknown_length:
            length = content.length
            break

    for other_content in it:
        if other_content.length is unknown_length:
            continue
        if other_content.length != length:
            raise ValueError(
                "cannot broadcast {} of length {} with {} of length {}{}".format(
                    type(content).__name__,
                    length,
                    type(other_content).__name__,
                    other_content.length,
                    in_function(options),
                )
            )


def all_same_offsets(backend: Backend, inputs: list) -> bool:
    index_nplike = backend.index_nplike

    offsets = None
    for x in inputs:
        if isinstance(x, ListOffsetArray):
            if offsets is None:
                offsets = x.offsets.raw(index_nplike)
            elif not index_nplike.array_equal(offsets, x.offsets.raw(index_nplike)):
                return False

        elif isinstance(x, ListArray):
            starts = x.starts.raw(index_nplike)
            stops = x.stops.raw(index_nplike)

            if not index_nplike.array_equal(starts[1:], stops[:-1]):
                return False

            elif offsets is None:
                offsets = index_nplike.empty(starts.shape[0] + 1, dtype=starts.dtype)
                if offsets.shape[0] == 1:
                    offsets[0] = 0
                else:
                    offsets[:-1] = starts
                    offsets[-1] = stops[-1]

            elif not index_nplike.array_equal(offsets[:-1], starts) or (
                stops.shape[0] != 0 and offsets[-1] != stops[-1]
            ):
                return False

        elif isinstance(x, RegularArray):
            if x.size == 0:
                my_offsets = index_nplike.empty(0, dtype=np.int64)
            else:
                my_offsets = index_nplike.arange(
                    0, x.content.length + 1, x.size, dtype=np.int64
                )

            if offsets is None:
                offsets = my_offsets
            elif not index_nplike.array_equal(offsets, my_offsets):
                return False

        elif isinstance(x, Content):
            return False

    return True


NO_PARAMETERS = Sentinel("NO_PARAMETERS", __name__)


class BroadcastParameterRule(str, enum.Enum):
    """Behaviour for parameter coalescence during broadcasting."""

    INTERSECT = "intersect"
    ONE_TO_ONE = "one_to_one"
    ALL_OR_NOTHING = "all_or_nothing"
    NONE = "none"


BroadcastParameterFactory: TypeAlias = Callable[
    [int], List[Union[Dict[str, Any], None]]
]


def _parameters_of(obj: Any, default: Any = NO_PARAMETERS) -> Any:
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
    inputs: Sequence,
) -> BroadcastParameterFactory:
    """
    Args:
        inputs: sequence of #ak.contents.Content or other objects

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "all or nothing rule":

    If the parameters of all the given contents are equal, then the first content's
    parameters are repeated, i.e. `[parameters, parameters, ...]`. Otherwise, a list
    of Nones is returned, i.e. `[None, None, ...]`.
    """
    input_parameters = [
        p for p in (_parameters_of(c) for c in inputs) if p is not NO_PARAMETERS
    ]

    parameters = None
    if len(input_parameters) > 0:
        # All parameters must match this first layout's parameters
        first_parameters = input_parameters[0]
        # Ensure all parameters match, or set parameters to None
        for other_parameters in input_parameters[1:]:
            if not parameters_are_equal(first_parameters, other_parameters):
                break
        else:
            parameters = first_parameters

    def apply(n_outputs: int) -> list[dict[str, Any] | None]:
        # NB: we don't make unique copies here, so let's hope everyone
        # is well-behaved downstream!
        return [parameters] * n_outputs

    return apply


def intersection_parameters_factory(
    inputs: Sequence,
) -> BroadcastParameterFactory:
    """
    Args:
        inputs: sequence of #ak.contents.Content or other objects

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "intersection rule":

    The intersection of `content._parameters.items()` for each content is computed.
    If any parameter dictionaries are None, then a list of Nones is returned, i.e.
    `[None, None, ...]`; otherwise, the computed parameter dictionary is repeated,
    i.e. `[parameters, parameters, ...]`.
    """
    input_parameters = [
        p for p in (_parameters_of(c) for c in inputs) if p is not NO_PARAMETERS
    ]

    intersected_parameters = None
    parameters_to_intersect = []
    # Build a list of set-like dict.items() views.
    # If we encounter None-parameters, then we stop early
    # as there can be no intersection.
    for parameters in input_parameters:
        if parameters_are_empty(parameters):
            break
        else:
            parameters_to_intersect.append(parameters)
    # Otherwise, build the intersected parameter dict
    else:
        intersected_parameters = functools.reduce(
            parameters_intersect, parameters_to_intersect
        )

    def apply(n_outputs: int) -> list[dict[str, Any] | None]:
        # NB: we don't make unique copies here, so let's hope everyone
        # is well-behaved downstream!
        return [intersected_parameters] * n_outputs

    return apply


def one_to_one_parameters_factory(
    inputs: Sequence,
) -> BroadcastParameterFactory:
    """
    Args:
        inputs: sequence of #ak.contents.Content or other objects

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using a "one-to-one rule":

    The requested number of outputs is compared against the length of the given
    `inputs`. If the two values match, then a list of parameter objects is returned,
    where each element of the returned list corresponds to the parameters of the
    content at the same position in the `inputs` sequence. If the length of the
    given contents does not match the requested list length, a ValueError is raised.
    """
    # Find the parameters of the inputs, with None values for non-Contents
    input_parameters = [_parameters_of(c, default=None) for c in inputs]

    def apply(n_outputs) -> list[dict[str, Any] | None]:
        if n_outputs != len(inputs):
            raise ValueError(
                "cannot follow one-to-one parameter broadcasting rule for actions "
                "which change the number of outputs."
            )
        return input_parameters

    return apply


def none_parameters_factory(
    inputs: Sequence,
) -> BroadcastParameterFactory:
    """
    Args:
        inputs: sequence of #ak.contents.Content or other objects

    Return a callable that creates an appropriately sized list of parameter objects.
    The parameter objects within this list are built using an "all or nothing rule":

    A list of Nones is returned, with a length corresponding to the requested number of
    outputs, i.e. `[None, None, ...]`.
    """

    def apply(n_outputs: int) -> list[dict[str, Any] | None]:
        return [None] * n_outputs

    return apply


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


def apply_step(
    backend: Backend,
    inputs: Sequence,
    action,
    depth: int,
    depth_context,
    lateral_context,
    behavior,
    options,
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
                    behavior,
                    options,
                )

    # Now all lengths must agree.
    checklength(contents, options)

    # Load the parameter broadcasting rule implementation
    rule = options["broadcast_parameters_rule"]
    try:
        parameters_factory_impl = BROADCAST_RULE_TO_FACTORY_IMPL[rule]
    except KeyError:
        raise ValueError(
            f"`broadcast_parameters_rule` should be one of {[str(x) for x in BroadcastParameterRule]}, "
            f"but this routine received `{rule}`"
        ) from None
    parameters_factory = parameters_factory_impl(inputs)

    # This whole function is one big switch statement.
    def broadcast_any_record():
        if not options["allow_records"]:
            raise ValueError(f"cannot broadcast records {in_function(options)}")

        fields, length, istuple = UNSET, UNSET, UNSET
        for x in contents:
            if x.is_record:
                if fields is UNSET:
                    fields = x.fields
                elif set(fields) != set(x.fields):
                    raise ValueError(
                        "cannot broadcast records because fields don't "
                        "match{}:\n    {}\n    {}".format(
                            in_function(options),
                            ", ".join(sorted(fields)),
                            ", ".join(sorted(x.fields)),
                        )
                    )
                if length is UNSET:
                    length = x.length
                elif length != x.length:
                    raise ValueError(
                        "cannot broadcast RecordArray of length {} "
                        "with RecordArray of length {}{}".format(
                            length, x.length, in_function(options)
                        )
                    )
                # Records win over tuples
                if istuple is UNSET or not x.is_tuple:
                    istuple = False

        outcontents, numoutputs = [], None
        for field in fields:
            outcontents.append(
                apply_step(
                    backend,
                    [x[field] if isinstance(x, RecordArray) else x for x in inputs],
                    action,
                    depth,
                    copy.copy(depth_context),
                    lateral_context,
                    behavior,
                    options,
                )
            )
            assert isinstance(outcontents[-1], tuple)
            if numoutputs is not None:
                assert numoutputs == len(outcontents[-1])
            numoutputs = len(outcontents[-1])

        return tuple(
            RecordArray(
                [x[i] for x in outcontents],
                None if istuple else fields,
                length,
                parameters=p,
            )
            for i, p in enumerate(parameters_factory(numoutputs))
        )

    def broadcast_any_list():
        index_nplike = backend.index_nplike
        # All regular?
        if all(x.is_regular or not x.is_list for x in contents):
            # Ensure all layouts have same length
            length = None
            for x in contents:
                if length is None:
                    length = x.length
                elif length is not unknown_length and x.length is not unknown_length:
                    assert length == x.length
            assert length is not None

            # Determine the size of the broadcast result
            dim_size = None
            for x in contents:
                if not x.is_regular:
                    continue

                # Any unknown length sets max_size to unknown
                if x.size is unknown_length:
                    dim_size = unknown_length
                # Any zero-length column triggers zero broadcasting
                elif x.size == 0:
                    dim_size = 0
                    break
                # Take first size as dim_size
                elif dim_size is None:
                    dim_size = x.size
                # If the dim_size is unknown, we can't compare
                elif dim_size is unknown_length:
                    continue
                else:
                    dim_size = max(dim_size, x.size)

            dimsize_greater_than_one_if_known = (
                dim_size is unknown_length or dim_size > 1
            )
            dimsize_known_to_be_zero = dim_size is not unknown_length and dim_size == 0

            # Build a broadcast index for size=1 contents
            size_one_carry_index = None
            for x in contents:
                if x.is_regular:
                    x_size_known_to_be_one = (
                        x.size is not unknown_length and x.size == 1
                    )
                    if dimsize_greater_than_one_if_known and x_size_known_to_be_one:
                        # For any (N, 1) array, we know we'll broadcast to (N, M) where M is maxsize
                        size_one_carry_index = Index64(
                            index_nplike.repeat(
                                index_nplike.arange(
                                    index_nplike.shape_item_as_index(x.length),
                                    dtype=np.int64,
                                ),
                                index_nplike.shape_item_as_index(dim_size),
                            ),
                            nplike=index_nplike,
                        )
                        break

            # Here we have three possible broadcasting outcomes (by precedence):
            # 1. any (exactly) size-0 content trims all other contents
            # 2. any (exactly) size-1 content broadcasts to the common length
            # 3. otherwise, recurse into the content as-is
            nextinputs = []
            for x in inputs:
                if isinstance(x, RegularArray):
                    x_size_known_to_be_one = (
                        x.size is not unknown_length and x.size == 1
                    )
                    # If dimsize is known to be exactly zero, all contents are zero length
                    if dimsize_known_to_be_zero:
                        nextinputs.append(x.content[:0])
                    # If we have a known size=1 content, then broadcast it to the dimension size
                    elif dimsize_greater_than_one_if_known and x_size_known_to_be_one:
                        nextinputs.append(
                            x.content[: x.length * x.size]._carry(
                                size_one_carry_index, allow_lazy=False
                            )
                        )
                    # Any unknown values or sizes are assumed to be correct as-is
                    elif (
                        dim_size is unknown_length
                        or x.size is unknown_length
                        or x.size == dim_size
                    ):
                        nextinputs.append(x.content[: x.length * x.size])
                    else:
                        raise ValueError(
                            "cannot broadcast RegularArray of size "
                            "{} with RegularArray of size {} {}".format(
                                x.size, dim_size, in_function(options)
                            )
                        )
                else:
                    nextinputs.append(x)

            outcontent = apply_step(
                backend,
                nextinputs,
                action,
                depth + 1,
                copy.copy(depth_context),
                lateral_context,
                behavior,
                options,
            )
            assert isinstance(outcontent, tuple)
            parameters = parameters_factory(len(outcontent))
            return tuple(
                RegularArray(x, dim_size, length, parameters=p)
                for x, p in zip(outcontent, parameters)
            )

        # Not all regular, but all same offsets?
        # Optimization: https://github.com/scikit-hep/awkward-1.0/issues/442
        elif index_nplike.known_data and all_same_offsets(backend, inputs):
            lencontent, offsets, starts, stops = None, None, None, None
            nextinputs = []

            for x in inputs:
                if isinstance(x, ListOffsetArray):
                    offsets = x.offsets
                    lencontent = index_nplike.index_as_shape_item(offsets[-1])
                    nextinputs.append(x.content[:lencontent])

                elif isinstance(x, ListArray):
                    starts, stops = x.starts, x.stops
                    if (starts.length is not unknown_length and starts.length == 0) or (
                        stops.length is not unknown_length and stops.length == 0
                    ):
                        nextinputs.append(x.content[:0])
                    else:
                        lencontent = index_nplike.index_as_shape_item(
                            index_nplike.max(stops)
                        )
                        nextinputs.append(x.content[:lencontent])
                elif isinstance(x, RegularArray):
                    nextinputs.append(x.content[: x.size * x.length])
                else:
                    nextinputs.append(x)

            outcontent = apply_step(
                backend,
                nextinputs,
                action,
                depth + 1,
                copy.copy(depth_context),
                lateral_context,
                behavior,
                options,
            )
            assert isinstance(outcontent, tuple)
            parameters = parameters_factory(len(outcontent))

            if isinstance(offsets, Index):
                return tuple(
                    ListOffsetArray(offsets, x, parameters=p).to_ListOffsetArray64(
                        False
                    )
                    for x, p in zip(outcontent, parameters)
                )
            elif isinstance(starts, Index) and isinstance(stops, Index):
                return tuple(
                    ListArray(starts, stops, x, parameters=p).to_ListOffsetArray64(
                        False
                    )
                    for x, p in zip(outcontent, parameters)
                )
            else:
                raise AssertionError(
                    "unexpected offsets, starts: {}, {}".format(
                        type(offsets), type(starts)
                    )
                )

        # General list-handling case: the offsets of each list may be different.
        else:
            # We have three possible cases here:
            # 1. all-string (nothing to do)
            # 2. mixed string-list (strings gain a dimension and broadcast to the non-string offsets)
            # 3. no strings (all lists broadcast to a single offsets)
            offsets_content = None
            all_content_strings = True
            input_is_string = []
            for x in inputs:
                if isinstance(x, Content):
                    content_is_string = x.parameter("__array__") in {
                        "string",
                        "bytestring",
                    }
                    # Don't try and take offsets from strings
                    if not content_is_string:
                        all_content_strings = False
                        # Take the offsets from the first irregular list
                        if x.is_list and not x.is_regular and offsets_content is None:
                            offsets_content = x
                    input_is_string.append(content_is_string)
                else:
                    input_is_string.append(False)

            # case (1): user getfunctions should exit before this gets called
            if all_content_strings:
                raise ValueError(
                    "cannot broadcast all strings: {}{}".format(
                        ", ".join(repr(type(x)) for x in inputs), in_function(options)
                    )
                )

            # Didn't find a ragged list, try any list!
            if offsets_content is None:
                for content in contents:
                    if content.is_list:
                        offsets_content = content
                        break
                else:
                    raise AssertionError("no list found in list branch!")

            offsets = offsets_content._compact_offsets64(True)

            nextinputs = []
            for x, x_is_string in zip(inputs, input_is_string):
                if x_is_string:
                    offsets_data = backend.index_nplike.asarray(offsets)
                    counts = offsets_data[1:] - offsets_data[:-1]
                    if ak._util.win or ak._util.bits32:
                        counts = index_nplike.astype(counts, dtype=np.int32)
                    parents = index_nplike.repeat(
                        index_nplike.arange(counts.size, dtype=counts.dtype), counts
                    )
                    nextinputs.append(
                        ak.contents.IndexedArray(
                            ak.index.Index64(parents, nplike=index_nplike), x
                        ).project()
                    )
                elif isinstance(x, listtypes):
                    nextinputs.append(x._broadcast_tooffsets64(offsets).content)
                # Handle implicit left-broadcasting (non-NumPy-like broadcasting).
                elif options["left_broadcast"] and isinstance(x, Content):
                    nextinputs.append(
                        RegularArray(x, 1, x.length)
                        ._broadcast_tooffsets64(offsets)
                        .content
                    )
                else:
                    nextinputs.append(x)

            outcontent = apply_step(
                backend,
                nextinputs,
                action,
                depth + 1,
                copy.copy(depth_context),
                lateral_context,
                behavior,
                options,
            )
            assert isinstance(outcontent, tuple)
            parameters = parameters_factory(len(outcontent))

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
        for x in inputs:
            if isinstance(x, optiontypes):
                nextinputs.append(x.project(nextmask))
            elif isinstance(x, Content):
                nextinputs.append(IndexedOptionArray(nextindex, x).project(nextmask))
            else:
                nextinputs.append(x)

        outcontent = apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            behavior,
            options,
        )
        assert isinstance(outcontent, tuple)
        parameters = parameters_factory(len(outcontent))
        return tuple(
            IndexedOptionArray.simplified(index, x, parameters=p)
            for x, p in zip(outcontent, parameters)
        )

    def broadcast_any_union():
        if not backend.nplike.known_data:
            # assert False
            union_num_contents = []
            length = None
            for x in contents:
                if x.is_union:
                    x._touch_data(recursive=False)
                    union_num_contents.append(len(x.contents))
                    if length is None:
                        length = x.length

            all_combos = list(
                itertools.product(*[range(x) for x in union_num_contents])
            )

            tags = backend.index_nplike.empty(length, dtype=np.int8)
            index = backend.index_nplike.empty(length, dtype=np.int64)
            numoutputs = None
            outcontents = []
            for combo in all_combos:
                nextinputs = []
                i = 0
                for x in inputs:
                    if isinstance(x, UnionArray):
                        nextinputs.append(x._contents[combo[i]])
                        i += 1
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
                        behavior,
                        options,
                    )
                )
                assert isinstance(outcontents[-1], tuple)
                if numoutputs is not None:
                    assert numoutputs == len(outcontents[-1])
                numoutputs = len(outcontents[-1])

            assert numoutputs is not None

        else:
            union_tags, union_num_contents, length = [], [], None
            for x in contents:
                if x.is_union:
                    tags = x.tags.raw(backend.index_nplike)
                    union_tags.append(tags)
                    union_num_contents.append(len(x.contents))
                    if tags.shape[0] is unknown_length:
                        continue

                    if length is None:
                        length = tags.shape[0]
                    elif length != tags.shape[0]:
                        raise ValueError(
                            "cannot broadcast UnionArray of length {} "
                            "with UnionArray of length {}{}".format(
                                length,
                                tags.shape[0],
                                in_function(options),
                            )
                        )
            assert length is not unknown_length

            # Stack all union tags
            combos = backend.index_nplike.stack(union_tags, axis=-1)
            # Build array of indices (c1, c2, c3, ..., cn) of contents in
            # (union 1, union 2, union 3, ..., union n)
            all_combos = backend.index_nplike.asarray(
                list(itertools.product(*[range(x) for x in union_num_contents]))
            )

            tags = backend.index_nplike.empty(length, dtype=np.int8)
            index = backend.index_nplike.empty(length, dtype=np.int64)
            numoutputs, outcontents = None, []
            for tag, combo in enumerate(all_combos):
                mask = backend.index_nplike.all(combos == combo, axis=-1)
                tags[mask] = tag
                index[mask] = backend.index_nplike.arange(
                    backend.index_nplike.count_nonzero(mask), dtype=np.int64
                )
                nextinputs = []
                i = 0
                for x in inputs:
                    if isinstance(x, UnionArray):
                        nextinputs.append(x[mask].project(combo[i]))
                        i += 1
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
                        behavior,
                        options,
                    )
                )
                assert isinstance(outcontents[-1], tuple)
                if numoutputs is None:
                    numoutputs = len(outcontents[-1])
                else:
                    assert numoutputs == len(outcontents[-1])

            assert numoutputs is not None

        parameters = parameters_factory(numoutputs)
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
        nextinputs = [x.project() if isinstance(x, IndexedArray) else x for x in inputs]
        return apply_step(
            backend,
            nextinputs,
            action,
            depth,
            copy.copy(depth_context),
            lateral_context,
            behavior,
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
            behavior,
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
            behavior,
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

        # Any list-types?
        elif any(x.is_list for x in contents):
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
        behavior=behavior,
        backend=backend,
        options=options,
    )

    if isinstance(result, tuple) and all(isinstance(x, Content) for x in result):
        return result
    elif result is None:
        return continuation()
    else:
        raise AssertionError(result)


def broadcast_and_apply(
    inputs,
    action,
    behavior,
    depth_context=None,
    lateral_context=None,
    allow_records=True,
    left_broadcast=True,
    right_broadcast=True,
    numpy_to_regular=False,
    regular_to_jagged=False,
    function_name=None,
    broadcast_parameters_rule=BroadcastParameterRule.INTERSECT,
):
    backend = backend_of(*inputs)
    isscalar = []
    out = apply_step(
        backend,
        broadcast_pack(inputs, isscalar),
        action,
        0,
        depth_context,
        lateral_context,
        behavior,
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
    return tuple(broadcast_unpack(x, isscalar, backend) for x in out)
