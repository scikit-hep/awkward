# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import inspect
import re
import sys
import os
import weakref
import warnings

try:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
except ImportError:
    from collections import Mapping
    from collections import MutableMapping

import awkward1.layout
import awkward1.partition
import awkward1.nplike


np = awkward1.nplike.NumpyMetadata.instance()


py27 = sys.version_info[0] < 3
py35 = sys.version_info[0] == 3 and sys.version_info[1] <= 5
win = os.name == "nt"


# to silence flake8 F821 errors
if py27:
    unicode = eval("unicode")
else:
    unicode = None


def exception_suffix(filename):
    line = ""
    if hasattr(sys, "_getframe"):
        line = "#L" + str(sys._getframe(1).f_lineno)
    filename = filename.replace("\\", "/")
    filename = "/src/awkward1/" + filename.split("awkward1/")[1]
    return ("\n\n(https://github.com/scikit-hep/awkward-1.0/blob/"
            + awkward1.__version__
            + filename
            + line
            + ")")


def deprecate(exception, version, date=None):
    if awkward1.deprecations_as_errors:
        raise exception
    else:
        if date is None:
            date = ""
        else:
            date = " (target date: " + date + ")"
        message = """In version {0}{1}, this will be an error.
(Set ak.deprecations_as_errors = True to get a stack trace now.)

{2}: {3}""".format(version, date, type(exception).__name__, str(exception))
        warnings.warn(message, DeprecationWarning)


virtualtypes = (awkward1.layout.VirtualArray,)

unknowntypes = (awkward1.layout.EmptyArray,)

indexedtypes = (
    awkward1.layout.IndexedArray32,
    awkward1.layout.IndexedArrayU32,
    awkward1.layout.IndexedArray64,
)

uniontypes = (
    awkward1.layout.UnionArray8_32,
    awkward1.layout.UnionArray8_U32,
    awkward1.layout.UnionArray8_64,
)

indexedoptiontypes = (
    awkward1.layout.IndexedOptionArray32,
    awkward1.layout.IndexedOptionArray64,
)

optiontypes = (
    awkward1.layout.IndexedOptionArray32,
    awkward1.layout.IndexedOptionArray64,
    awkward1.layout.ByteMaskedArray,
    awkward1.layout.BitMaskedArray,
    awkward1.layout.UnmaskedArray,
)

listtypes = (
    awkward1.layout.RegularArray,
    awkward1.layout.ListArray32,
    awkward1.layout.ListArrayU32,
    awkward1.layout.ListArray64,
    awkward1.layout.ListOffsetArray32,
    awkward1.layout.ListOffsetArrayU32,
    awkward1.layout.ListOffsetArray64,
)

recordtypes = (awkward1.layout.RecordArray,)


class Behavior(Mapping):
    def __init__(self, defaults, overrides):
        self.defaults = defaults
        if overrides is None:
            self.overrides = {}
        else:
            self.overrides = overrides

    def __getitem__(self, where):
        try:
            return self.overrides[where]
        except KeyError:
            try:
                return self.defaults[where]
            except KeyError:
                return None

    def items(self):
        for n, x in self.overrides.items():
            yield n, x
        for n, x in self.defaults.items():
            if n not in self.overrides:
                yield n, x

    def __iter__(self):
        for n, x in self.items():
            yield n

    def __len__(self):
        return len(set(self.defaults) | set(self.overrides))


def arrayclass(layout, behavior):
    layout = awkward1.partition.first(layout)
    behavior = Behavior(awkward1.behavior, behavior)
    arr = layout.parameter("__array__")
    if isinstance(arr, str) or (py27 and isinstance(arr, unicode)):
        cls = behavior[arr]
        if isinstance(cls, type) and issubclass(cls, awkward1.highlevel.Array):
            return cls
    rec = layout.parameter("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        cls = behavior[".", rec]
        if isinstance(cls, type) and issubclass(cls, awkward1.highlevel.Array):
            return cls
    deeprec = layout.purelist_parameter("__record__")
    if isinstance(deeprec, str) or (py27 and isinstance(deeprec, unicode)):
        cls = behavior["*", deeprec]
        if isinstance(cls, type) and issubclass(cls, awkward1.highlevel.Array):
            return cls
    return awkward1.highlevel.Array


def custom_broadcast(layout, behavior):
    layout = awkward1.partition.first(layout)
    behavior = Behavior(awkward1.behavior, behavior)
    custom = layout.parameter("__array__")
    if not (isinstance(custom, str) or (py27 and isinstance(custom, unicode))):
        custom = layout.parameter("__record__")
    if not (isinstance(custom, str) or (py27 and isinstance(custom, unicode))):
        custom = layout.purelist_parameter("__record__")
    if isinstance(custom, str) or (py27 and isinstance(custom, unicode)):
        for key, fcn in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] == "__broadcast__"
                and key[1] == custom
            ):
                return fcn
    return None


def numba_array_typer(layouttype, behavior):
    behavior = Behavior(awkward1.behavior, behavior)
    arr = layouttype.parameters.get("__array__")
    if isinstance(arr, str) or (py27 and isinstance(arr, unicode)):
        typer = behavior["__numba_typer__", arr]
        if callable(typer):
            return typer
    rec = layouttype.parameters.get("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        typer = behavior["__numba_typer__", ".", rec]
        if callable(typer):
            return typer
    deeprec = layouttype.parameters.get("__record__")
    if isinstance(deeprec, str) or (py27 and isinstance(deeprec, unicode)):
        typer = behavior["__numba_typer__", "*", deeprec]
        if callable(typer):
            return typer
    return None


def numba_array_lower(layouttype, behavior):
    behavior = Behavior(awkward1.behavior, behavior)
    arr = layouttype.parameters.get("__array__")
    if isinstance(arr, str) or (py27 and isinstance(arr, unicode)):
        lower = behavior["__numba_lower__", arr]
        if callable(lower):
            return lower
    rec = layouttype.parameters.get("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        lower = behavior["__numba_lower__", ".", rec]
        if callable(lower):
            return lower
    deeprec = layouttype.parameters.get("__record__")
    if isinstance(deeprec, str) or (py27 and isinstance(deeprec, unicode)):
        lower = behavior["__numba_lower__", "*", deeprec]
        if callable(lower):
            return lower
    return None


def recordclass(layout, behavior):
    layout = awkward1.partition.first(layout)
    behavior = Behavior(awkward1.behavior, behavior)
    rec = layout.parameter("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        cls = behavior[rec]
        if isinstance(cls, type) and issubclass(cls, awkward1.highlevel.Record):
            return cls
    return awkward1.highlevel.Record


def typestrs(behavior):
    behavior = Behavior(awkward1.behavior, behavior)
    out = {}
    for key, typestr in behavior.items():
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "__typestr__"
            and (isinstance(key[1], str) or (py27 and isinstance(key[1], unicode)))
            and (isinstance(typestr, str) or (py27 and isinstance(typestr, unicode)))
        ):
            out[key[1]] = typestr
    return out


def numba_record_typer(layouttype, behavior):
    behavior = Behavior(awkward1.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        typer = behavior["__numba_typer__", rec]
        if callable(typer):
            return typer
    return None


def numba_record_lower(layouttype, behavior):
    behavior = Behavior(awkward1.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        lower = behavior["__numba_lower__", rec]
        if callable(lower):
            return lower
    return None


def overload(behavior, signature):
    if not any(s is None for s in signature):
        behavior = Behavior(awkward1.behavior, behavior)
        for key, custom in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == len(signature)
                and key[0] == signature[0]
                and all(
                    k == s or (
                        isinstance(k, type)
                        and isinstance(s, type)
                        and issubclass(s, k)
                    ) for k, s in zip(key[1:], signature[1:])
                )
            ):
                return custom


def numba_attrs(layouttype, behavior):
    behavior = Behavior(awkward1.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        for key, typer in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 3
                and key[0] == "__numba_typer__"
                and key[1] == rec
            ):
                lower = behavior["__numba_lower__", key[1], key[2]]
                yield key[2], typer, lower


def numba_methods(layouttype, behavior):
    behavior = Behavior(awkward1.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
        for key, typer in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 4
                and key[0] == "__numba_typer__"
                and key[1] == rec
                and key[3] == ()
            ):
                lower = behavior["__numba_lower__", key[1], key[2], ()]
                yield key[2], typer, lower


def numba_unaryops(unaryop, left, behavior):
    import awkward1._connect._numba.layout

    behavior = Behavior(awkward1.behavior, behavior)
    done = False

    if isinstance(left, awkward1._connect._numba.layout.ContentType):
        left = left.parameters.get("__record__")
        if not (isinstance(left, str) or (py27 and isinstance(left, unicode))):
            done = True

    if not done:
        for key, typer in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 3
                and key[0] == "__numba_typer__"
                and key[1] == unaryop
                and key[2] == left
            ):
                lower = behavior["__numba_lower__", key[1], key[2]]
                yield typer, lower


def numba_binops(binop, left, right, behavior):
    import awkward1._connect._numba.layout

    behavior = Behavior(awkward1.behavior, behavior)
    done = False

    if isinstance(left, awkward1._connect._numba.layout.ContentType):
        left = left.parameters.get("__record__")
        if not (isinstance(left, str) or (py27 and isinstance(left, unicode))):
            done = True

    if isinstance(right, awkward1._connect._numba.layout.ContentType):
        right = right.parameters.get("__record__")
        if not isinstance(right, str) and not (py27 and isinstance(right, unicode)):
            done = True

    if not done:
        for key, typer in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 4
                and key[0] == "__numba_typer__"
                and key[1] == left
                and key[2] == binop
                and key[3] == right
            ):
                lower = behavior["__numba_lower__", key[1], key[2], key[3]]
                yield typer, lower


def behaviorof(*arrays):
    behavior = None
    for x in arrays[::-1]:
        if (
            isinstance(
                x,
                (
                    awkward1.highlevel.Array,
                    awkward1.highlevel.Record,
                    awkward1.highlevel.ArrayBuilder,
                ),
            )
            and x.behavior is not None
        ):
            if behavior is None:
                behavior = dict(x.behavior)
            else:
                behavior.update(x.behavior)
    return behavior


def wrap(content, behavior):
    import awkward1.highlevel

    if isinstance(
        content, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
    ):
        return awkward1.highlevel.Array(content, behavior=behavior, kernels=None)

    elif isinstance(content, awkward1.layout.Record):
        return awkward1.highlevel.Record(content, behavior=behavior, kernels=None)

    else:
        return content


def extra(args, kwargs, defaults):
    out = []
    for i in range(len(defaults)):
        name, default = defaults[i]
        if i < len(args):
            out.append(args[i])
        elif name in kwargs:
            out.append(kwargs[name])
        else:
            out.append(default)
    return out


def key2index(keys, key):
    if keys is None:
        attempt = None
    else:
        try:
            attempt = keys.index(key)
        except ValueError:
            attempt = None

    if attempt is None:
        m = key2index._pattern.match(key)
        if m is not None:
            attempt = m.group(0)

    if attempt is None:
        raise ValueError(
            "key {0} not found in record".format(repr(key))
            + exception_suffix(__file__)
        )
    else:
        return attempt


key2index._pattern = re.compile(r"^[1-9][0-9]*$")


def completely_flatten(array):
    if isinstance(array, awkward1.partition.PartitionedArray):
        out = []
        for partition in array.partitions:
            for outi in completely_flatten(partition):
                out.append(outi)
        return tuple(out)

    elif isinstance(array, virtualtypes):
        return completely_flatten(array.array)

    elif isinstance(array, unknowntypes):
        return (awkward1.nplike.of(array).array([], dtype=np.bool_),)

    elif isinstance(array, indexedtypes):
        return completely_flatten(array.project())

    elif isinstance(array, uniontypes):
        out = []
        for i in range(array.numcontents):
            out.append(completely_flatten(array.project(i)))
        return tuple(out)

    elif isinstance(array, optiontypes):
        return completely_flatten(array.project())

    elif isinstance(array, listtypes):
        return completely_flatten(array.flatten(axis=1))

    elif isinstance(array, recordtypes):
        out = []
        for i in range(array.numfields):
            out.extend(completely_flatten(array.field(i)))
        return tuple(out)

    elif isinstance(array, awkward1.layout.NumpyArray):
        return (awkward1.nplike.of(array).asarray(array),)

    else:
        raise RuntimeError(
            "cannot completely flatten: {0}".format(type(array))
            + exception_suffix(__file__)
        )


def broadcast_and_apply(inputs, getfunction, behavior, allow_records=True):
    def checklength(inputs):
        length = len(inputs[0])
        for x in inputs[1:]:
            if len(x) != length:
                raise ValueError(
                    "cannot broadcast {0} of length {1} with {2} of "
                    "length {3}".format(
                        type(inputs[0]).__name__, length, type(x).__name__, len(x)
                    )
                    + exception_suffix(__file__)
                )

    def all_same_offsets(nplike, inputs):
        offsets = None
        for x in inputs:
            if isinstance(x, (
                awkward1.layout.ListOffsetArray32,
                awkward1.layout.ListOffsetArrayU32,
                awkward1.layout.ListOffsetArray64,
            )):
                if offsets is None:
                    offsets = nplike.asarray(x.offsets)
                elif not nplike.array_equal(offsets, nplike.asarray(x.offsets)):
                    return False
            elif isinstance(x, (
                awkward1.layout.ListArray32,
                awkward1.layout.ListArrayU32,
                awkward1.layout.ListArray64,
            )):
                starts = nplike.asarray(x.starts)
                stops = nplike.asarray(x.stops)
                if not nplike.array_equal(starts[1:], stops[:-1]):
                    return False
                if offsets is None:
                    offsets = nplike.empty(len(starts) + 1, dtype=starts.dtype)
                    if len(offsets) == 1:
                        offsets[0] = 0
                    else:
                        offsets[:-1] = starts
                        offsets[-1] = stops[-1]
                elif (
                    not nplike.array_equal(offsets[:-1], starts) or
                    (len(stops) !=0 and offsets[-1] != stops[-1])
                ):
                    return False
            elif isinstance(x, awkward1.layout.RegularArray):
                my_offsets = nplike.arange(0, len(x.content), x.size)
                if offsets is None:
                    offsets = my_offsets
                elif not nplike.array_equal(offsets, my_offsets):
                    return False
            elif isinstance(x, awkward1.layout.Content):
                return False
        else:
            return True

    def apply(inputs, depth):
        nplike = awkward1.nplike.of(*inputs)

        # handle implicit right-broadcasting (i.e. NumPy-like)
        if any(isinstance(x, listtypes) for x in inputs):
            maxdepth = max(
                x.purelist_depth
                for x in inputs
                if isinstance(x, awkward1.layout.Content)
            )

            if maxdepth > 0 and all(
                x.purelist_isregular
                for x in inputs
                if isinstance(x, awkward1.layout.Content)
            ):
                nextinputs = []
                for x in inputs:
                    if isinstance(x, awkward1.layout.Content):
                        while x.purelist_depth < maxdepth:
                            x = awkward1.layout.RegularArray(x, 1)
                    nextinputs.append(x)
                if any(x is not y for x, y in zip(inputs, nextinputs)):
                    return apply(nextinputs, depth)

        # now all lengths must agree
        checklength([x for x in inputs if isinstance(x, awkward1.layout.Content)])

        function = getfunction(inputs, depth)

        # the rest of this is one switch statement
        if function is not None:
            return function()

        elif any(isinstance(x, virtualtypes) for x in inputs):
            return apply(
                [x if not isinstance(x, virtualtypes) else x.array for x in inputs],
                depth,
            )

        elif any(isinstance(x, unknowntypes) for x in inputs):
            return apply(
                [
                    x
                    if not isinstance(x, unknowntypes)
                    else awkward1.layout.NumpyArray(nplike.array([], dtype=np.bool_))
                    for x in inputs
                ],
                depth,
            )

        elif any(
            isinstance(x, awkward1.layout.NumpyArray) and x.ndim > 1 for x in inputs
        ):
            return apply(
                [
                    x
                    if not (isinstance(x, awkward1.layout.NumpyArray) and x.ndim > 1)
                    else x.toRegularArray()
                    for x in inputs
                ],
                depth,
            )

        elif any(isinstance(x, indexedtypes) for x in inputs):
            return apply(
                [x if not isinstance(x, indexedtypes) else x.project() for x in inputs],
                depth,
            )

        elif any(isinstance(x, uniontypes) for x in inputs):
            tagslist = []
            length = None
            for x in inputs:
                if isinstance(x, uniontypes):
                    tagslist.append(nplike.asarray(x.tags))
                    if length is None:
                        length = len(tagslist[-1])
                    elif length != len(tagslist[-1]):
                        raise ValueError(
                            "cannot broadcast UnionArray of length {0} "
                            "with UnionArray of length {1}".format(
                                length, len(tagslist[-1])
                            )
                            + exception_suffix(__file__)
                        )

            combos = nplike.stack(tagslist, axis=-1)
            combos = combos.view(
                [(str(i), combos.dtype) for i in range(len(tagslist))]
            ).reshape(length)

            tags = nplike.empty(length, dtype=np.int8)
            index = nplike.empty(length, dtype=np.int64)
            outcontents = []
            for tag, combo in enumerate(nplike.unique(combos)):
                mask = combos == combo
                tags[mask] = tag
                index[mask] = nplike.arange(nplike.count_nonzero(mask))
                nextinputs = []
                numoutputs = None
                i = 0
                for x in inputs:
                    if isinstance(x, uniontypes):
                        nextinputs.append(x[mask].project(combo[str(i)]))
                        i += 1
                    elif isinstance(x, awkward1.layout.Content):
                        nextinputs.append(x[mask])
                    else:
                        nextinputs.append(x)
                outcontents.append(apply(nextinputs, depth))
                assert isinstance(outcontents[-1], tuple)
                if numoutputs is not None:
                    assert numoutputs == len(outcontents[-1])
                numoutputs = len(outcontents[-1])

            tags = awkward1.layout.Index8(tags)
            index = awkward1.layout.Index64(index)

            return tuple(
                awkward1.layout.UnionArray8_64(
                    tags, index, [x[i] for x in outcontents]
                ).simplify()
                for i in range(numoutputs)
            )

        elif any(isinstance(x, optiontypes) for x in inputs):
            mask = None
            for x in inputs:
                if isinstance(x, optiontypes):
                    m = nplike.asarray(x.bytemask()).view(np.bool_)
                    if mask is None:
                        mask = m
                    else:
                        nplike.bitwise_or(mask, m, out=mask)

            nextmask = awkward1.layout.Index8(mask.view(np.int8))
            index = nplike.full(len(mask), -1, dtype=np.int64)
            index[~mask] = nplike.arange(
                len(mask) - nplike.count_nonzero(mask), dtype=np.int64
            )
            index = awkward1.layout.Index64(index)
            if any(not isinstance(x, optiontypes) for x in inputs):
                nextindex = nplike.arange(len(mask), dtype=np.int64)
                nextindex[mask] = -1
                nextindex = awkward1.layout.Index64(nextindex)

            nextinputs = []
            for x in inputs:
                if isinstance(x, optiontypes):
                    nextinputs.append(x.project(nextmask))
                elif isinstance(x, awkward1.layout.Content):
                    nextinputs.append(
                        awkward1.layout.IndexedOptionArray64(nextindex, x).project(
                            nextmask
                        )
                    )
                else:
                    nextinputs.append(x)

            outcontent = apply(nextinputs, depth)
            assert isinstance(outcontent, tuple)
            return tuple(
                awkward1.layout.IndexedOptionArray64(index, x).simplify()
                for x in outcontent
            )

        elif any(isinstance(x, listtypes) for x in inputs):
            if all(
                isinstance(x, awkward1.layout.RegularArray)
                or not isinstance(x, listtypes)
                for x in inputs
            ):
                maxsize = max(
                    [
                        x.size
                        for x in inputs
                        if isinstance(x, awkward1.layout.RegularArray)
                    ]
                )
                for x in inputs:
                    if isinstance(x, awkward1.layout.RegularArray):
                        if maxsize > 1 and x.size == 1:
                            tmpindex = awkward1.layout.Index64(
                                nplike.repeat(
                                    nplike.arange(len(x), dtype=np.int64), maxsize
                                )
                            )
                nextinputs = []
                for x in inputs:
                    if isinstance(x, awkward1.layout.RegularArray):
                        if maxsize > 1 and x.size == 1:
                            nextinputs.append(
                                awkward1.layout.IndexedArray64(
                                    tmpindex, x.content[: len(x) * x.size]
                                ).project()
                            )
                        elif x.size == maxsize:
                            nextinputs.append(x.content[: len(x) * x.size])
                        else:
                            raise ValueError(
                                "cannot broadcast RegularArray of size "
                                "{0} with RegularArray of size {1}".format(
                                    x.size, maxsize
                                )
                                + exception_suffix(__file__)
                            )
                    else:
                        nextinputs.append(x)

                outcontent = apply(nextinputs, depth + 1)
                assert isinstance(outcontent, tuple)

                return tuple(
                    awkward1.layout.RegularArray(x, maxsize) for x in outcontent
                )

            elif not all_same_offsets(nplike, inputs):
                fcns = [
                    custom_broadcast(x, behavior)
                    if isinstance(x, awkward1.layout.Content)
                    else None
                    for x in inputs
                ]

                first, secondround = None, False
                for x, fcn in zip(inputs, fcns):
                    if (
                        isinstance(x, listtypes)
                        and not isinstance(x, awkward1.layout.RegularArray)
                        and fcn is None
                    ):
                        first = x
                        break

                if first is None:
                    secondround = True
                    for x in inputs:
                        if isinstance(x, listtypes) and not isinstance(
                            x, awkward1.layout.RegularArray
                        ):
                            first = x
                            break

                offsets = first.compact_offsets64(True)

                nextinputs = []
                for x, fcn in zip(inputs, fcns):
                    if callable(fcn) and not secondround:
                        nextinputs.append(fcn(x, offsets))
                    elif isinstance(x, listtypes):
                        nextinputs.append(x.broadcast_tooffsets64(offsets).content)
                    # handle implicit left-broadcasting (unlike NumPy)
                    elif isinstance(x, awkward1.layout.Content):
                        nextinputs.append(
                            awkward1.layout.RegularArray(x, 1)
                            .broadcast_tooffsets64(offsets)
                            .content
                        )
                    else:
                        nextinputs.append(x)

                outcontent = apply(nextinputs, depth + 1)
                assert isinstance(outcontent, tuple)

                return tuple(
                    awkward1.layout.ListOffsetArray64(offsets, x) for x in outcontent
                )

            else:
                lencontent, offsets, starts, stops = None, None, None, None
                nextinputs = []

                for x in inputs:
                    if isinstance(x, (
                        awkward1.layout.ListOffsetArray32,
                        awkward1.layout.ListOffsetArrayU32,
                        awkward1.layout.ListOffsetArray64,
                    )):
                        offsets = x.offsets
                        lencontent = offsets[-1]
                        nextinputs.append(x.content[:lencontent])

                    elif isinstance(x, (
                        awkward1.layout.ListArray32,
                        awkward1.layout.ListArrayU32,
                        awkward1.layout.ListArray64,
                    )):
                        starts, stops = x.starts, x.stops
                        if len(starts) == 0 or len(stops) == 0:
                            nextinputs.append(x.content[:0])
                        else:
                            lencontent = nplike.max(stops)
                            nextinputs.append(x.content[:lencontent])

                    else:
                        nextinputs.append(x)

                outcontent = apply(nextinputs, depth + 1)

                if isinstance(offsets, awkward1.layout.Index32):
                    return tuple(
                        awkward1.layout.ListOffsetArray32(offsets, x) for x in outcontent
                    )
                elif isinstance(offsets, awkward1.layout.IndexU32):
                    return tuple(
                        awkward1.layout.ListOffsetArrayU32(offsets, x) for x in outcontent
                    )
                elif isinstance(offsets, awkward1.layout.Index64):
                    return tuple(
                        awkward1.layout.ListOffsetArray64(offsets, x) for x in outcontent
                    )
                elif isinstance(starts, awkward1.layout.Index32):
                    return tuple(
                        awkward1.layout.ListArray32(starts, stops, x) for x in outcontent
                    )
                elif isinstance(starts, awkward1.layout.IndexU32):
                    return tuple(
                        awkward1.layout.ListArrayU32(starts, stops, x) for x in outcontent
                    )
                elif isinstance(starts, awkward1.layout.Index64):
                    return tuple(
                        awkward1.layout.ListArray64(starts, stops, x) for x in outcontent
                    )
                else:
                    raise AssertionError(
                        "unexpected offsets, starts: {0} {1}".format(
                            type(offsets), type(starts)
                        ) + exception_suffix(__file__)
                    )

        elif any(isinstance(x, recordtypes) for x in inputs):
            if not allow_records:
                exception = ValueError(
                    "cannot broadcast: {0}".format(", ".join(repr(type(x)) for x in inputs))
                    + exception_suffix(__file__)
                )
                deprecate(exception, "1.0.0", "2020-12-01")

            keys = None
            length = None
            istuple = True
            for x in inputs:
                if isinstance(x, recordtypes):
                    if keys is None:
                        keys = x.keys()
                    elif set(keys) != set(x.keys()):
                        raise ValueError(
                            "cannot broadcast records because keys don't "
                            "match:\n    {0}\n    {1}".format(
                                ", ".join(sorted(keys)), ", ".join(sorted(x.keys()))
                            )
                            + exception_suffix(__file__)
                        )
                    if length is None:
                        length = len(x)
                    elif length != len(x):
                        raise ValueError(
                            "cannot broadcast RecordArray of length {0} "
                            "with RecordArray of length {1}".format(length, len(x))
                            + exception_suffix(__file__)
                        )
                    if not x.istuple:
                        istuple = False

            outcontents = []
            numoutputs = None
            for key in keys:
                outcontents.append(
                    apply(
                        [
                            x if not isinstance(x, recordtypes) else x[key]
                            for x in inputs
                        ],
                        depth,
                    )
                )
                assert isinstance(outcontents[-1], tuple)
                if numoutputs is not None:
                    assert numoutputs == len(outcontents[-1])
                numoutputs = len(outcontents[-1])
            return tuple(
                awkward1.layout.RecordArray(
                    [x[i] for x in outcontents], None if istuple else keys, length
                )
                for i in range(numoutputs)
            )

        else:
            raise ValueError(
                "cannot broadcast: {0}".format(", ".join(repr(type(x)) for x in inputs))
                + exception_suffix(__file__)
            )

    if any(isinstance(x, awkward1.partition.PartitionedArray) for x in inputs):
        purelist_isregular = True
        purelist_depths = set()
        for x in inputs:
            if isinstance(
                x, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
            ):
                if not x.purelist_isregular:
                    purelist_isregular = False
                    break
                purelist_depths.add(x.purelist_depth)

        if purelist_isregular and len(purelist_depths) > 1:
            nextinputs = []
            for x in inputs:
                if isinstance(x, awkward1.partition.PartitionedArray):
                    nextinputs.append(x.toContent())
                else:
                    nextinputs.append(x)

            isscalar = []
            out = apply(broadcast_pack(nextinputs, isscalar), 0)
            assert isinstance(out, tuple)
            return tuple(broadcast_unpack(x, isscalar) for x in out)

        else:
            sample = None
            for x in inputs:
                if isinstance(x, awkward1.partition.PartitionedArray):
                    sample = x
                    break
            nextinputs = awkward1.partition.partition_as(sample, inputs)

            outputs = []
            for part_inputs in awkward1.partition.iterate(
                sample.numpartitions, nextinputs
            ):
                isscalar = []
                part = apply(broadcast_pack(part_inputs, isscalar), 0)
                assert isinstance(part, tuple)
                outputs.append(tuple(broadcast_unpack(x, isscalar) for x in part))

            out = ()
            for i in range(len(part)):
                out = out + (
                    awkward1.partition.IrregularlyPartitionedArray(
                        [x[i] for x in outputs]
                    ),
                )
            return out

    else:
        isscalar = []
        out = apply(broadcast_pack(inputs, isscalar), 0)
        assert isinstance(out, tuple)
        return tuple(broadcast_unpack(x, isscalar) for x in out)


def broadcast_pack(inputs, isscalar):
    maxlen = -1
    for x in inputs:
        if isinstance(x, awkward1.layout.Content):
            maxlen = max(maxlen, len(x))
    if maxlen < 0:
        maxlen = 1

    nextinputs = []
    for x in inputs:
        if isinstance(x, awkward1.layout.Record):
            index = awkward1.nplike.of(*inputs).full(maxlen, x.at, dtype=np.int64)
            nextinputs.append(awkward1.layout.RegularArray(x.array[index], maxlen))
            isscalar.append(True)
        elif isinstance(x, awkward1.layout.Content):
            nextinputs.append(awkward1.layout.RegularArray(x, len(x)))
            isscalar.append(False)
        else:
            nextinputs.append(x)
            isscalar.append(True)

    return nextinputs


def broadcast_unpack(x, isscalar):
    if all(isscalar):
        if len(x) == 0:
            return x.getitem_nothing().getitem_nothing()
        else:
            return x[0][0]
    else:
        if len(x) == 0:
            return x.getitem_nothing()
        else:
            return x[0]


def recursively_apply(layout, getfunction, args=(), depth=1, keep_parameters=True):
    custom = getfunction(layout, depth, *args)
    if custom is not None:
        return custom()

    elif isinstance(layout, awkward1.partition.PartitionedArray):
        return awkward1.partition.IrregularlyPartitionedArray(
            [
                recursively_apply(x, getfunction, args, depth, keep_parameters)
                for x in layout.partitions
            ]
        )

    elif isinstance(layout, awkward1.layout.NumpyArray):
        if keep_parameters:
            return layout
        else:
            return awkward1.layout.NumpyArray(
                awkward1.nplike.of(layout).asarray(layout), layout.identities, None
            )

    elif isinstance(layout, awkward1.layout.EmptyArray):
        if keep_parameters:
            return layout
        else:
            return awkward1.layout.EmptyArray(layout.identities, None)

    elif isinstance(layout, awkward1.layout.RegularArray):
        return awkward1.layout.RegularArray(
            recursively_apply(
                layout.content, getfunction, args, depth + 1, keep_parameters
            ),
            layout.size,
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.ListArray32):
        return awkward1.layout.ListArray32(
            layout.starts,
            layout.stops,
            recursively_apply(
                layout.content, getfunction, args, depth + 1, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.ListArrayU32):
        return awkward1.layout.ListArrayU32(
            layout.starts,
            layout.stops,
            recursively_apply(
                layout.content, getfunction, args, depth + 1, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.ListArray64):
        return awkward1.layout.ListArray64(
            layout.starts,
            layout.stops,
            recursively_apply(
                layout.content, getfunction, args, depth + 1, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.ListOffsetArray32):
        return awkward1.layout.ListOffsetArray32(
            layout.offsets,
            recursively_apply(
                layout.content, getfunction, args, depth + 1, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.ListOffsetArrayU32):
        return awkward1.layout.ListOffsetArrayU32(
            layout.offsets,
            recursively_apply(
                layout.content, getfunction, args, depth + 1, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.ListOffsetArray64):
        return awkward1.layout.ListOffsetArray64(
            layout.offsets,
            recursively_apply(
                layout.content, getfunction, args, depth + 1, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.IndexedArray32):
        return awkward1.layout.IndexedArray32(
            layout.index,
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.IndexedArrayU32):
        return awkward1.layout.IndexedArrayU32(
            layout.index,
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.IndexedArray64):
        return awkward1.layout.IndexedArray64(
            layout.index,
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.IndexedOptionArray32):
        return awkward1.layout.IndexedOptionArray32(
            layout.index,
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.IndexedOptionArray64):
        return awkward1.layout.IndexedOptionArray64(
            layout.index,
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.ByteMaskedArray):
        return awkward1.layout.ByteMaskedArray(
            layout.mask,
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.valid_when,
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.BitMaskedArray):
        return awkward1.layout.BitMaskedArray(
            layout.mask,
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.valid_when,
            len(layout),
            layout.lsb_order,
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.UnmaskedArray):
        return awkward1.layout.UnmaskedArray(
            recursively_apply(
                layout.content, getfunction, args, depth, keep_parameters
            ),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.RecordArray):
        return awkward1.layout.RecordArray(
            [
                recursively_apply(x, getfunction, args, depth, keep_parameters)
                for x in layout.contents
            ],
            layout.recordlookup,
            len(layout),
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.Record):
        return awkward1.layout.Record(
            recursively_apply(layout.array, getfunction, args, depth, keep_parameters),
            layout.at,
        )

    elif isinstance(layout, awkward1.layout.UnionArray8_32):
        return awkward1.layout.UnionArray8_32(
            layout.tags,
            layout.index,
            [
                recursively_apply(x, getfunction, args, depth, keep_parameters)
                for x in layout.contents
            ],
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.UnionArray8_U32):
        return awkward1.layout.UnionArray8_U32(
            layout.tags,
            layout.index,
            [
                recursively_apply(x, getfunction, args, depth, keep_parameters)
                for x in layout.contents
            ],
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.UnionArray8_64):
        return awkward1.layout.UnionArray8_64(
            layout.tags,
            layout.index,
            [
                recursively_apply(x, getfunction, args, depth, keep_parameters)
                for x in layout.contents
            ],
            layout.identities,
            layout.parameters if keep_parameters else None,
        )

    elif isinstance(layout, awkward1.layout.VirtualArray):
        return recursively_apply(
            layout.array, getfunction, args, depth, keep_parameters
        )

    else:
        raise AssertionError(
            "unrecognized Content type: {0}".format(type(layout))
            + exception_suffix(__file__)
        )


def recursive_walk(layout, apply, args=(), depth=1, materialize=False):
    apply(layout, depth, *args)

    if isinstance(layout, awkward1.partition.PartitionedArray):
        for x in layout.partitions:
            recursive_walk(x, apply, args, depth, materialize)

    elif isinstance(layout, awkward1.layout.NumpyArray):
        pass

    elif isinstance(layout, awkward1.layout.EmptyArray):
        pass

    elif isinstance(layout, (
        awkward1.layout.RegularArray,
        awkward1.layout.ListArray32,
        awkward1.layout.ListArrayU32,
        awkward1.layout.ListArray64,
        awkward1.layout.ListOffsetArray32,
        awkward1.layout.ListOffsetArrayU32,
        awkward1.layout.ListOffsetArray64,
    )):
        recursive_walk(layout.content, apply, args, depth + 1, materialize)

    elif isinstance(layout, (
        awkward1.layout.IndexedArray32,
        awkward1.layout.IndexedArrayU32,
        awkward1.layout.IndexedArray64,
        awkward1.layout.IndexedOptionArray32,
        awkward1.layout.IndexedOptionArray64,
        awkward1.layout.ByteMaskedArray,
        awkward1.layout.BitMaskedArray,
        awkward1.layout.UnmaskedArray,
    )):
        recursive_walk(layout.content, apply, args, depth, materialize)

    elif isinstance(layout, (
        awkward1.layout.RecordArray,
        awkward1.layout.UnionArray8_32,
        awkward1.layout.UnionArray8_U32,
        awkward1.layout.UnionArray8_64,
    )):
        for x in layout.contents:
            recursive_walk(x, apply, args, depth, materialize)

    elif isinstance(layout, awkward1.layout.Record):
        recursive_walk(layout.array, apply, args, depth, materialize)

    elif isinstance(layout, awkward1.layout.VirtualArray):
        if materialize:
            recursive_walk(layout.array, apply, args, depth, materialize)

    else:
        raise AssertionError(
            "unrecognized Content type: {0}".format(type(layout))
            + exception_suffix(__file__)
        )


def find_caches(layout):
    if isinstance(layout, awkward1.partition.PartitionedArray):
        seen = set()
        mutablemappings = []
        for partition in layout.partitions:
            for cache in partition.caches:
                x = cache.mutablemapping
                if id(x) not in seen:
                    seen.add(id(x))
                    mutablemappings.append(x)
    else:
        mutablemappings = []
        for cache in layout.caches:
            x = cache.mutablemapping
            for y in mutablemappings:
                if x is y:
                    break
            else:
                mutablemappings.append(x)

    return tuple(mutablemappings)


def highlevel_type(layout, behavior, isarray):
    if isarray:
        return awkward1.types.ArrayType(layout.type(typestrs(behavior)), len(layout))
    else:
        return layout.type(typestrs(behavior))


_is_identifier = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def minimally_touching_string(limit_length, layout, behavior):
    import awkward1.layout

    if isinstance(layout, awkward1.layout.Record):
        layout = layout.array[layout.at : layout.at + 1]

    if len(layout) == 0:
        return "[]"

    def forward(x, space, brackets=True, wrap=True, stop=None):
        done = False
        if wrap and isinstance(
            x, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
        ):
            cls = arrayclass(x, behavior)
            if cls is not awkward1.highlevel.Array:
                y = cls(x, behavior=behavior)
                if "__repr__" in type(y).__dict__:
                    yield space + repr(y)
                    done = True
        if wrap and isinstance(x, awkward1.layout.Record):
            cls = recordclass(x, behavior)
            if cls is not awkward1.highlevel.Record:
                y = cls(x, behavior=behavior)
                if "__repr__" in type(y).__dict__:
                    yield space + repr(y)
                    done = True
        if not done:
            if isinstance(
                x, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
            ):
                if brackets:
                    yield space + "["
                sp = ""
                for i in range(len(x) if stop is None else stop):
                    for token in forward(x[i], sp):
                        yield token
                    sp = ", "
                if brackets:
                    yield "]"
            elif isinstance(x, awkward1.layout.Record) and x.istuple:
                yield space + "("
                sp = ""
                for i in range(x.numfields):
                    key = sp
                    for token in forward(x[str(i)], ""):
                        yield key + token
                        key = ""
                    sp = ", "
                yield ")"
            elif isinstance(x, awkward1.layout.Record):
                yield space + "{"
                sp = ""
                for k in x.keys():
                    if _is_identifier.match(k) is None:
                        kk = repr(k)
                        if kk.startswith("u"):
                            kk = kk[1:]
                    else:
                        kk = k
                    key = sp + kk + ": "
                    for token in forward(x[k], ""):
                        yield key + token
                        key = ""
                    sp = ", "
                yield "}"
            elif isinstance(x, (float, np.floating)):
                yield space + "{0:.3g}".format(x)
            else:
                yield space + repr(x)

    def backward(x, space, brackets=True, wrap=True, stop=-1):
        done = False
        if wrap and isinstance(
            x, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
        ):
            cls = arrayclass(x, behavior)
            if cls is not awkward1.highlevel.Array:
                y = cls(x, behavior=behavior)
                if "__repr__" in type(y).__dict__:
                    yield repr(y) + space
                    done = True
        if wrap and isinstance(x, awkward1.layout.Record):
            cls = recordclass(x, behavior)
            if cls is not awkward1.highlevel.Record:
                y = cls(x, behavior=behavior)
                if "__repr__" in type(y).__dict__:
                    yield repr(y) + space
                    done = True
        if not done:
            if isinstance(
                x, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
            ):
                if brackets:
                    yield "]" + space
                sp = ""
                for i in range(len(x) - 1, stop, -1):
                    for token in backward(x[i], sp):
                        yield token
                    sp = ", "
                if brackets:
                    yield "["
            elif isinstance(x, awkward1.layout.Record) and x.istuple:
                yield ")" + space
                for i in range(x.numfields - 1, -1, -1):
                    last = None
                    for token in backward(x[str(i)], ""):
                        if last is not None:
                            yield last
                        last = token
                    if last is not None:
                        yield last
                    if i != 0:
                        yield ", "
                yield "("
            elif isinstance(x, awkward1.layout.Record):
                yield "}" + space
                keys = x.keys()
                for i in range(len(keys) - 1, -1, -1):
                    last = None
                    for token in backward(x[keys[i]], ""):
                        if last is not None:
                            yield last
                        last = token
                    if _is_identifier.match(keys[i]) is None:
                        kk = repr(keys[i])
                        if kk.startswith("u"):
                            kk = kk[1:]
                    else:
                        kk = keys[i]
                    if last is not None:
                        yield kk + ": " + last
                    if i != 0:
                        yield ", "
                yield "{"
            elif isinstance(x, (float, np.floating)):
                yield "{0:.3g}".format(x) + space
            else:
                yield repr(x) + space

    def forever(iterable):
        for token in iterable:
            yield token
        while True:
            yield None

    halfway = len(layout) // 2
    left, right = ["["], ["]"]
    leftlen, rightlen = 1, 1
    leftgen = forever(forward(layout, "", brackets=False, wrap=False, stop=halfway))
    rightgen = forever(
        backward(layout, "", brackets=False, wrap=False, stop=halfway - 1)
    )
    while True:
        lft = next(leftgen)
        rgt = next(rightgen)

        if lft is not None:
            if (
                leftlen
                + rightlen
                + len(lft)
                + (2 if lft is None and rgt is None else 6)
                > limit_length
            ):
                break
            left.append(lft)
            leftlen += len(lft)

        if rgt is not None:
            if (
                leftlen
                + rightlen
                + len(rgt)
                + (2 if lft is None and rgt is None else 6)
                > limit_length
            ):
                break
            right.append(rgt)
            rightlen += len(rgt)

        if lft is None and rgt is None:
            break

    while len(left) > 1 and (
        left[-1] == "["
        or left[-1] == ", ["
        or left[-1] == "{"
        or left[-1] == ", {"
        or left[-1] == ", "
    ):
        left.pop()
        lft = ""
    while len(right) > 1 and (
        right[-1] == "]"
        or right[-1] == "], "
        or right[-1] == "}"
        or right[-1] == "}, "
        or right[-1] == ", "
    ):
        right.pop()
        rgt = ""
    if lft is None and rgt is None:
        if left == ["["]:
            return "[" + "".join(reversed(right)).lstrip(" ")
        else:
            return (
                "".join(left).rstrip(" ") + ", " + "".join(reversed(right)).lstrip(" ")
            )
    else:
        if left == ["["] and right == ["]"]:
            return "[...]"
        elif left == ["["]:
            return "[... " + "".join(reversed(right)).lstrip(" ")
        else:
            return (
                "".join(left).rstrip(" ")
                + ", ... "
                + "".join(reversed(right)).lstrip(" ")
            )


class MappingProxy(MutableMapping):
    """
    A type suitable for use with layout.ArrayCache.

    This can be used to wrap plain dict instances if need be,
    since they are not able to be weak referenced.
    """
    @classmethod
    def maybe_wrap(cls, mapping):
        if type(mapping) is dict:
            return cls(mapping)
        return mapping

    def __init__(self, base):
        self.base = base

    def __repr__(self):
        return repr(self.base)

    def __getitem__(self, key):
        return self.base[key]

    def __setitem__(self, key, value):
        self.base[key] = value

    def __delitem__(self, key):
        del self.base[key]

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)


def make_union(tags, index, contents, identities, parameters):
    if isinstance(index, awkward1.layout.Index32):
        return awkward1.layout.UnionArray8_32(
            tags, index, contents, identities, parameters
        )
    elif isinstance(index, awkward1.layout.IndexU32):
        return awkward1.layout.UnionArray8_U32(
            tags, index, contents, identities, parameters
        )
    elif isinstance(index, awkward1.layout.Index64):
        return awkward1.layout.UnionArray8_64(
            tags, index, contents, identities, parameters
        )
    else:
        raise AssertionError(index)


def union_to_record(unionarray, anonymous):
    nplike = awkward1.nplike.of(unionarray)

    contents = []
    for layout in unionarray.contents:
        if isinstance(layout, virtualtypes):
            contents.append(layout.array)
        elif isinstance(layout, indexedtypes):
            contents.append(layout.project())
        elif isinstance(layout, uniontypes):
            contents.append(union_to_record(layout, anonymous))
        elif isinstance(layout, optiontypes):
            contents.append(awkward1.operations.structure.fill_none(
                layout, np.nan, highlevel=False
            ))
        else:
            contents.append(layout)

    if not any(isinstance(x, awkward1.layout.RecordArray) for x in contents):
        return make_union(
            unionarray.tags,
            unionarray.index,
            contents,
            unionarray.identities,
            unionarray.parameters,
        )

    else:
        seen = set()
        all_names = []
        for layout in contents:
            if isinstance(layout, awkward1.layout.RecordArray):
                for key in layout.keys():
                    if key not in seen:
                        seen.add(key)
                        all_names.append(key)
            else:
                if anonymous not in seen:
                    seen.add(anonymous)
                    all_names.append(anonymous)

        missingarray = awkward1.layout.IndexedOptionArray64(
            awkward1.layout.Index64(nplike.full(len(unionarray), -1, dtype=np.int64)),
            awkward1.layout.EmptyArray(),
        )

        all_fields = []
        for name in all_names:
            union_contents = []
            for layout in contents:
                if isinstance(layout, awkward1.layout.RecordArray):
                    for key in layout.keys():
                        if name == key:
                            union_contents.append(layout.field(key))
                            break
                    else:
                        union_contents.append(missingarray)
                else:
                    if name == anonymous:
                        union_contents.append(layout)
                    else:
                        union_contents.append(missingarray)

            all_fields.append(
                make_union(
                    unionarray.tags,
                    unionarray.index,
                    union_contents,
                    unionarray.identities,
                    unionarray.parameters,
                ).simplify()
            )

        return awkward1.layout.RecordArray(all_fields, all_names, len(unionarray))
