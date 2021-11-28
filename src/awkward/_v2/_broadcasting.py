# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import copy
import itertools

import awkward as ak
from awkward._v2.contents.content import Content  # noqa: F401
from awkward._v2.contents.emptyarray import EmptyArray  # noqa: F401
from awkward._v2.contents.numpyarray import NumpyArray  # noqa: F401
from awkward._v2.contents.regulararray import RegularArray  # noqa: F401
from awkward._v2.contents.listarray import ListArray  # noqa: F401
from awkward._v2.contents.listoffsetarray import ListOffsetArray  # noqa: F401
from awkward._v2.contents.recordarray import RecordArray  # noqa: F401
from awkward._v2.contents.indexedarray import IndexedArray  # noqa: F401
from awkward._v2.contents.indexedoptionarray import IndexedOptionArray  # noqa: F401
from awkward._v2.contents.bytemaskedarray import ByteMaskedArray  # noqa: F401
from awkward._v2.contents.bitmaskedarray import BitMaskedArray  # noqa: F401
from awkward._v2.contents.unmaskedarray import UnmaskedArray  # noqa: F401
from awkward._v2.contents.unionarray import UnionArray  # noqa: F401
from awkward._v2.record import Record  # noqa: F401
from awkward._v2.index import (
    Index,  # noqa: F401
    Index8,  # noqa: F401
    #    IndexU8,  # noqa: F401
    #    Index32,  # noqa: F401
    #    IndexU32,  # noqa: F401
    Index64,  # noqa: F401
)

np = ak.nplike.NumpyMetadata.instance()

optiontypes = (IndexedOptionArray, ByteMaskedArray, BitMaskedArray, UnmaskedArray)
listtypes = (ListOffsetArray, ListArray, RegularArray)


def broadcast_pack(inputs, isscalar):
    maxlen = -1
    for x in inputs:
        if isinstance(x, Content):
            maxlen = max(maxlen, len(x))
    if maxlen < 0:
        maxlen = 1

    nextinputs = []
    for x in inputs:
        if isinstance(x, Record):
            index = ak.nplike.of(*inputs).full(maxlen, x.at, dtype=np.int64)
            nextinputs.append(RegularArray(x.array[index], maxlen, 1))
            isscalar.append(True)
        elif isinstance(x, Content):
            nextinputs.append(RegularArray(x, len(x), 1))
            isscalar.append(False)
        else:
            nextinputs.append(x)
            isscalar.append(True)

    return nextinputs


def broadcast_unpack(x, isscalar):
    if all(isscalar):
        if len(x) == 0:
            return x._getitem_nothing()._getitem_nothing()
        else:
            return x[0][0]
    else:
        if len(x) == 0:
            return x._getitem_nothing()
        else:
            return x[0]


def in_function(options):
    if options["function_name"] is None:
        return ""
    else:
        return "in " + options["function_name"]


def checklength(inputs, options):
    length = len(inputs[0])
    for x in inputs[1:]:
        if len(x) != length:
            raise ValueError(
                "cannot broadcast {0} of length {1} with {2} of length {3}{4}".format(
                    type(inputs[0]).__name__,
                    length,
                    type(x).__name__,
                    len(x),
                    in_function(options),
                )
            )


def all_same_offsets(nplike, inputs):
    offsets = None
    for x in inputs:
        if isinstance(x, ListOffsetArray):
            if offsets is None:
                offsets = x.offsets.to(nplike)
            elif not nplike.array_equal(offsets, x.offsets.to(nplike)):
                return False

        elif isinstance(x, ListArray):
            starts = x.starts.to(nplike)
            stops = x.stops.to(nplike)

            if not nplike.array_equal(starts[1:], stops[:-1]):
                return False

            elif offsets is None:
                offsets = nplike.empty(len(starts) + 1, dtype=starts.dtype)
                if len(offsets) == 1:
                    offsets[0] = 0
                else:
                    offsets[:-1] = starts
                    offsets[-1] = stops[-1]

            elif not nplike.array_equal(offsets[:-1], starts) or (
                len(stops) != 0 and offsets[-1] != stops[-1]
            ):
                return False

        elif isinstance(x, RegularArray):
            if x.size == 0:
                my_offsets = nplike.empty(0, dtype=np.int64)
            else:
                my_offsets = nplike.arange(0, len(x.content), x.size)

            if offsets is None:
                offsets = my_offsets
            elif not nplike.array_equal(offsets, my_offsets):
                return False

        elif isinstance(x, Content):
            return False

    return True


def apply_step(
    nplike, inputs, action, depth, depth_context, lateral_context, behavior, options
):
    # This happens when descending anyway, but setting the option does it before action.
    if options["numpy_to_regular"] and any(
        isinstance(x, NumpyArray) and x.data.ndim != 1 for x in inputs
    ):
        inputs = [
            x.toRegularArray() if isinstance(x, NumpyArray) else x for x in inputs
        ]

    # Rare that any function would want this, but some do.
    if options["regular_to_jagged"] and any(
        isinstance(x, RegularArray) for x in inputs
    ):
        inputs = [
            x.toListOffsetArray64(False) if isinstance(x, RegularArray) else x
            for x in inputs
        ]

    # Handle implicit right-broadcasting (NumPy-like broadcasting).
    if options["right_broadcast"] and any(isinstance(x, listtypes) for x in inputs):
        maxdepth = max(x.purelist_depth for x in inputs if isinstance(x, Content))

        if maxdepth > 0 and all(
            x.purelist_isregular for x in inputs if isinstance(x, Content)
        ):
            nextinputs = []
            for obj in inputs:
                if isinstance(obj, Content):
                    while obj.purelist_depth < maxdepth:
                        obj = RegularArray(obj, 1, len(obj))
                nextinputs.append(obj)
            if any(x is not y for x, y in zip(inputs, nextinputs)):
                return apply_step(
                    nplike,
                    nextinputs,
                    action,
                    depth,
                    depth_context,
                    lateral_context,
                    behavior,
                    options,
                )

    # Now all lengths must agree.
    checklength([x for x in inputs if isinstance(x, Content)], options)

    # This whole function is one big switch statement.
    def continuation():
        # Any EmptyArrays?
        if any(isinstance(x, EmptyArray) for x in inputs):
            nextinputs = [
                x.toNumpyArray(np.float64, nplike) if isinstance(x, EmptyArray) else x
                for x in inputs
            ]
            return apply_step(
                nplike,
                nextinputs,
                action,
                depth,
                copy.copy(depth_context),
                lateral_context,
                behavior,
                options,
            )

        # Any NumpyArrays with ndim != 1?
        elif any(isinstance(x, NumpyArray) and x.data.ndim != 1 for x in inputs):
            nextinputs = [
                x.toRegularArray() if isinstance(x, NumpyArray) else x for x in inputs
            ]
            return apply_step(
                nplike,
                nextinputs,
                action,
                depth,
                copy.copy(depth_context),
                lateral_context,
                behavior,
                options,
            )

        # Any IndexedArrays?
        elif any(isinstance(x, IndexedArray) for x in inputs):
            nextinputs = [
                x.project() if isinstance(x, IndexedArray) else x for x in inputs
            ]
            return apply_step(
                nplike,
                nextinputs,
                action,
                depth,
                copy.copy(depth_context),
                lateral_context,
                behavior,
                options,
            )

        # Any UnionArrays?
        elif any(isinstance(x, UnionArray) for x in inputs):
            tagslist, numtags, length = [], [], None
            for x in inputs:
                if isinstance(x, UnionArray):
                    tagslist.append(x.tags.to(nplike))
                    numtags.append(len(x.contents))
                    if length is None:
                        length = len(tagslist[-1])
                    elif length != len(tagslist[-1]):
                        raise ValueError(
                            "cannot broadcast UnionArray of length {0} "
                            "with UnionArray of length {1}{2}".format(
                                length, len(tagslist[-1]), in_function(options)
                            )
                        )

            combos = nplike.stack(tagslist, axis=-1)

            all_combos = nplike.array(
                list(itertools.product(*[range(x) for x in numtags])),
                dtype=[(str(i), combos.dtype) for i in range(len(tagslist))],
            )

            combos = combos.view(
                [(str(i), combos.dtype) for i in range(len(tagslist))]
            ).reshape(length)

            tags = nplike.empty(length, dtype=np.int8)
            index = nplike.empty(length, dtype=np.int64)
            numoutputs, outcontents = None, []
            for tag, combo in enumerate(all_combos):
                mask = combos == combo
                tags[mask] = tag
                index[mask] = nplike.arange(nplike.count_nonzero(mask))
                nextinputs = []
                i = 0
                for x in inputs:
                    if isinstance(x, UnionArray):
                        nextinputs.append(x[mask].project(combo[str(i)]))
                        i += 1
                    elif isinstance(x, Content):
                        nextinputs.append(x[mask])
                    else:
                        nextinputs.append(x)
                outcontents.append(
                    apply_step(
                        nplike,
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

            return tuple(
                UnionArray(
                    Index8(tags), Index64(index), [x[i] for x in outcontents]
                ).simplify_uniontype()
                for i in range(numoutputs)
            )

        # Any option-types?
        elif any(isinstance(x, optiontypes) for x in inputs):
            mask = None
            for x in inputs:
                if isinstance(x, optiontypes):
                    m = x.mask_as_bool(valid_when=False, nplike=nplike)
                    if mask is None:
                        mask = m
                    else:
                        mask = nplike.bitwise_or(mask, m, out=mask)

            nextmask = Index8(mask.view(np.int8))
            index = nplike.full(len(mask), -1, dtype=np.int64)
            index[~mask] = nplike.arange(
                len(mask) - nplike.count_nonzero(mask), dtype=np.int64
            )
            index = Index64(index)
            if any(not isinstance(x, optiontypes) for x in inputs):
                nextindex = nplike.arange(len(mask), dtype=np.int64)
                nextindex[mask] = -1
                nextindex = Index64(nextindex)

            nextinputs = []
            for x in inputs:
                if isinstance(x, optiontypes):
                    nextinputs.append(x.project(nextmask))
                elif isinstance(x, Content):
                    nextinputs.append(
                        IndexedOptionArray(nextindex, x).project(nextmask)
                    )
                else:
                    nextinputs.append(x)

            outcontent = apply_step(
                nplike,
                nextinputs,
                action,
                depth,
                copy.copy(depth_context),
                lateral_context,
                behavior,
                options,
            )
            assert isinstance(outcontent, tuple)
            return tuple(
                IndexedOptionArray(index, x).simplify_optiontype() for x in outcontent
            )

        # Any list-types?
        elif any(isinstance(x, listtypes) for x in inputs):
            # All regular?
            if all(
                isinstance(x, RegularArray) or not isinstance(x, listtypes)
                for x in inputs
            ):
                maxsize = max([x.size for x in inputs if isinstance(x, RegularArray)])
                for x in inputs:
                    if isinstance(x, RegularArray):
                        if maxsize > 1 and x.size == 1:
                            tmpindex = Index64(
                                nplike.repeat(
                                    nplike.arange(len(x), dtype=np.int64), maxsize
                                )
                            )

                nextinputs = []
                for x in inputs:

                    if isinstance(x, RegularArray):
                        if maxsize > 1 and x.size == 1:
                            nextinputs.append(
                                IndexedArray(
                                    tmpindex, x.content[: len(x) * x.size]
                                ).project()
                            )
                        elif x.size == maxsize:
                            nextinputs.append(x.content[: len(x) * x.size])
                        else:
                            raise ValueError(
                                "cannot broadcast RegularArray of size "
                                "{0} with RegularArray of size {1}{2}".format(
                                    x.size, maxsize, in_function(options)
                                )
                            )
                    else:
                        nextinputs.append(x)

                maxlen = max(len(x) for x in nextinputs if isinstance(x, Content))
                outcontent = apply_step(
                    nplike,
                    nextinputs,
                    action,
                    depth + 1,
                    copy.copy(depth_context),
                    lateral_context,
                    behavior,
                    options,
                )
                assert isinstance(outcontent, tuple)
                return tuple(RegularArray(x, maxsize, maxlen) for x in outcontent)

            # Not all regular, but all same offsets?
            # Optimization: https://github.com/scikit-hep/awkward-1.0/issues/442
            elif all_same_offsets(nplike, inputs):
                lencontent, offsets, starts, stops = None, None, None, None
                nextinputs = []

                for x in inputs:
                    if isinstance(x, ListOffsetArray):
                        offsets = x.offsets
                        lencontent = offsets[-1]
                        nextinputs.append(x.content[:lencontent])

                    elif isinstance(x, ListArray):
                        starts, stops = x.starts, x.stops
                        if len(starts) == 0 or len(stops) == 0:
                            nextinputs.append(x.content[:0])
                        else:
                            lencontent = nplike.max(stops)
                            nextinputs.append(x.content[:lencontent])

                    else:
                        nextinputs.append(x)

                outcontent = apply_step(
                    nplike,
                    nextinputs,
                    action,
                    depth + 1,
                    copy.copy(depth_context),
                    lateral_context,
                    behavior,
                    options,
                )
                assert isinstance(outcontent, tuple)

                if isinstance(offsets, Index):
                    return tuple(ListOffsetArray(offsets, x) for x in outcontent)
                elif isinstance(starts, Index) and isinstance(stops, Index):
                    return tuple(ListArray(starts, stops, x) for x in outcontent)
                else:
                    raise AssertionError(
                        "unexpected offsets, starts: {0}, {1}".format(
                            type(offsets), type(starts)
                        )
                    )

            # General list-handling case: the offsets of each list may be different.
            else:
                fcns = [
                    ak._v2._util.custom_broadcast(x, behavior)
                    if isinstance(x, Content)
                    else None
                    for x in inputs
                ]

                first, secondround = None, False
                for x, fcn in zip(inputs, fcns):
                    if (
                        isinstance(x, listtypes)
                        and not isinstance(x, RegularArray)
                        and fcn is None
                    ):
                        first = x
                        break

                if first is None:
                    secondround = True
                    for x in inputs:
                        if isinstance(x, listtypes) and not isinstance(x, RegularArray):
                            first = x
                            break

                offsets = first._compact_offsets64(True)

                nextinputs = []
                for x, fcn in zip(inputs, fcns):
                    if callable(fcn) and not secondround:
                        nextinputs.append(fcn(x, offsets))
                    elif isinstance(x, listtypes):
                        nextinputs.append(x._broadcast_tooffsets64(offsets).content)

                    # Handle implicit left-broadcasting (non-NumPy-like broadcasting).
                    elif options["left_broadcast"] and isinstance(x, Content):
                        nextinputs.append(
                            RegularArray(x, 1, len(x))
                            ._broadcast_tooffsets64(offsets)
                            .content
                        )
                    else:
                        nextinputs.append(x)

                outcontent = apply_step(
                    nplike,
                    nextinputs,
                    action,
                    depth + 1,
                    copy.copy(depth_context),
                    lateral_context,
                    behavior,
                    options,
                )
                assert isinstance(outcontent, tuple)

                return tuple(ListOffsetArray(offsets, x) for x in outcontent)

        # Any RecordArrays?
        elif any(isinstance(x, RecordArray) for x in inputs):
            if not options["allow_records"]:
                raise ValueError(
                    "cannot broadcast records{0}".format(in_function(options))
                )

            fields, length, istuple = None, None, True
            for x in inputs:
                if isinstance(x, RecordArray):
                    if fields is None:
                        fields = x.fields
                    elif set(fields) != set(x.fields):
                        raise ValueError(
                            "cannot broadcast records because fields don't "
                            "match{0}:\n    {1}\n    {2}".format(
                                in_function(options),
                                ", ".join(sorted(fields)),
                                ", ".join(sorted(x.fields)),
                            )
                        )
                    if length is None:
                        length = len(x)
                    elif length != len(x):
                        raise ValueError(
                            "cannot broadcast RecordArray of length {0} "
                            "with RecordArray of length {1}{2}".format(
                                length, len(x), in_function(options)
                            )
                        )
                    if not x.is_tuple:
                        istuple = False
            outcontents, numoutputs = [], None
            for field in fields:
                outcontents.append(
                    apply_step(
                        nplike,
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
                    [x[i] for x in outcontents], None if istuple else fields, length
                )
                for i in range(numoutputs)
            )

        else:
            raise ValueError(
                "cannot broadcast: {0}{1}".format(
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
        nplike=nplike,
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
):
    isscalar = []
    out = apply_step(
        ak.nplike.of(*inputs),
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
        },
    )
    assert isinstance(out, tuple)
    return tuple(broadcast_unpack(x, isscalar) for x in out)
