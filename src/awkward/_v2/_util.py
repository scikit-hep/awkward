# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# First, transition all the _v2 code to start using implementations in this file.
# Then build up the high-level replacements.

from __future__ import absolute_import

# import re
# import sys
import os

# import os.path
# import warnings
# import itertools
import numbers

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

win = os.name == "nt"
bits32 = ak.nplike.numpy.iinfo(np.intp).bits == 32

# matches include/awkward/common.h
kMaxInt8 = 127  # 2**7  - 1
kMaxUInt8 = 255  # 2**8  - 1
kMaxInt32 = 2147483647  # 2**31 - 1
kMaxUInt32 = 4294967295  # 2**32 - 1
kMaxInt64 = 9223372036854775806  # 2**63 - 2: see below
kSliceNone = kMaxInt64 + 1  # for Slice::none()
kMaxLevels = 48


def in_module(obj, modulename):
    m = type(obj).__module__
    return m == modulename or m.startswith(modulename + ".")


def is_file_path(x):
    try:
        return os.path.isfile(x)
    except ValueError:
        return False


def isint(x):
    return isinstance(
        x, (int, numbers.Integral, np.integer, ak._v2._typetracer.Interval)
    ) and not isinstance(x, (bool, np.bool_))


def isnum(x):
    return isinstance(x, (int, float, numbers.Real, np.number)) and not isinstance(
        x, (bool, np.bool_)
    )


def isstr(x):
    return isinstance(x, str)


###############################################################################

# # Enable warnings for the Awkward package
# warnings.filterwarnings("default", module="awkward.*")


# def deprecate(
#     message,
#     version,
#     date=None,
#     will_be="an error",
#     category=DeprecationWarning,
#     stacklevel=2,
# ):
#     if date is None:
#         date = ""
#     else:
#         date = " (target date: " + date + ")"
#     warning = """In version {0}{1}, this will be {2}.

# To raise these warnings as errors (and get stack traces to find out where they're called), run

#     import warnings
#     warnings.filterwarnings("error", module="awkward.*")

# after the first `import awkward` or use `@pytest.mark.filterwarnings("error:::awkward.*")` in pytest.

# Issue: {3}.""".format(
#         version, date, will_be, message
#     )
#     warnings.warn(warning, category, stacklevel=stacklevel + 1)


# # Sentinel object for catching pass-through values
# class MISSING(object):
#     pass


# virtualtypes = (ak._v2.contents.VirtualArray,)

# unknowntypes = (ak._v2.contents.EmptyArray,)

# indexedtypes = (
#     ak._v2.contents.IndexedArray32,
#     ak._v2.contents.IndexedArrayU32,
#     ak._v2.contents.IndexedArray64,
# )

# uniontypes = (
#     ak._v2.contents.UnionArray8_32,
#     ak._v2.contents.UnionArray8_U32,
#     ak._v2.contents.UnionArray8_64,
# )

# indexedoptiontypes = (
#     ak._v2.contents.IndexedOptionArray32,
#     ak._v2.contents.IndexedOptionArray64,
# )

# optiontypes = (
#     ak._v2.contents.IndexedOptionArray32,
#     ak._v2.contents.IndexedOptionArray64,
#     ak._v2.contents.ByteMaskedArray,
#     ak._v2.contents.BitMaskedArray,
#     ak._v2.contents.UnmaskedArray,
# )

# listtypes = (
#     ak._v2.contents.RegularArray,
#     ak._v2.contents.ListArray32,
#     ak._v2.contents.ListArrayU32,
#     ak._v2.contents.ListArray64,
#     ak._v2.contents.ListOffsetArray32,
#     ak._v2.contents.ListOffsetArrayU32,
#     ak._v2.contents.ListOffsetArray64,
# )

# recordtypes = (ak._v2.contents.RecordArray,)


# def regularize_path(path):
#     """
#     Converts pathlib Paths into plain string paths (for all versions of Python).
#     """
#     is_path = False

#     if isinstance(path, getattr(os, "PathLike", ())):
#         is_path = True
#         path = os.fspath(path)

#     elif hasattr(path, "__fspath__"):
#         is_path = True
#         path = path.__fspath__()

#     elif path.__class__.__module__ == "pathlib":
#         import pathlib

#         if isinstance(path, pathlib.Path):
#             is_path = True
#             path = str(path)

#     return is_path, path


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
        for x in self.keys():
            yield x

    def __len__(self):
        return len(set(self.defaults) | set(self.overrides))


def arrayclass(layout, behavior):
    behavior = Behavior(ak.behavior, behavior)
    arr = layout.parameter("__array__")
    if isstr(arr):
        cls = behavior[arr]
        if isinstance(cls, type) and issubclass(cls, ak._v2.highlevel.Array):
            return cls
    rec = layout.parameter("__record__")
    if isstr(rec):
        cls = behavior[".", rec]
        if isinstance(cls, type) and issubclass(cls, ak._v2.highlevel.Array):
            return cls
    deeprec = layout.purelist_parameter("__record__")
    if isstr(deeprec):
        cls = behavior["*", deeprec]
        if isinstance(cls, type) and issubclass(cls, ak._v2.highlevel.Array):
            return cls
    return ak._v2.highlevel.Array


# def custom_cast(obj, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     for key, fcn in behavior.items():
#         if (
#             isinstance(key, tuple)
#             and len(key) == 2
#             and key[0] == "__cast__"
#             and isinstance(obj, key[1])
#         ):
#             return fcn
#     return None


# def custom_broadcast(layout, behavior):
#     layout = ak.partition.first(layout)
#     behavior = Behavior(ak.behavior, behavior)
#     custom = layout.parameter("__array__")
#     if not (isinstance(custom, str) or (py27 and isinstance(custom, unicode))):
#         custom = layout.parameter("__record__")
#     if not (isinstance(custom, str) or (py27 and isinstance(custom, unicode))):
#         custom = layout.purelist_parameter("__record__")
#     if isinstance(custom, str) or (py27 and isinstance(custom, unicode)):
#         for key, fcn in behavior.items():
#             if (
#                 isinstance(key, tuple)
#                 and len(key) == 2
#                 and key[0] == "__broadcast__"
#                 and key[1] == custom
#             ):
#                 return fcn
#     return None


# def numba_array_typer(layouttype, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     arr = layouttype.parameters.get("__array__")
#     if isinstance(arr, str) or (py27 and isinstance(arr, unicode)):
#         typer = behavior["__numba_typer__", arr]
#         if callable(typer):
#             return typer
#     rec = layouttype.parameters.get("__record__")
#     if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
#         typer = behavior["__numba_typer__", ".", rec]
#         if callable(typer):
#             return typer
#     deeprec = layouttype.parameters.get("__record__")
#     if isinstance(deeprec, str) or (py27 and isinstance(deeprec, unicode)):
#         typer = behavior["__numba_typer__", "*", deeprec]
#         if callable(typer):
#             return typer
#     return None


# def numba_array_lower(layouttype, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     arr = layouttype.parameters.get("__array__")
#     if isinstance(arr, str) or (py27 and isinstance(arr, unicode)):
#         lower = behavior["__numba_lower__", arr]
#         if callable(lower):
#             return lower
#     rec = layouttype.parameters.get("__record__")
#     if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
#         lower = behavior["__numba_lower__", ".", rec]
#         if callable(lower):
#             return lower
#     deeprec = layouttype.parameters.get("__record__")
#     if isinstance(deeprec, str) or (py27 and isinstance(deeprec, unicode)):
#         lower = behavior["__numba_lower__", "*", deeprec]
#         if callable(lower):
#             return lower
#     return None


def recordclass(layout, behavior):
    behavior = Behavior(ak.behavior, behavior)
    rec = layout.parameter("__record__")
    if isstr(rec):
        cls = behavior[rec]
        if isinstance(cls, type) and issubclass(cls, ak._v2.highlevel.Record):
            return cls
    return ak._v2.highlevel.Record


# def typestrs(behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     out = {}
#     for key, typestr in behavior.items():
#         if (
#             isinstance(key, tuple)
#             and len(key) == 2
#             and key[0] == "__typestr__"
#             and (isinstance(key[1], str) or (py27 and isinstance(key[1], unicode)))
#             and (isinstance(typestr, str) or (py27 and isinstance(typestr, unicode)))
#         ):
#             out[key[1]] = typestr
#     return out


# def gettypestr(parameters, typestrs):
#     if parameters is not None:
#         record = parameters.get("__record__")
#         if record is not None:
#             typestr = typestrs.get(record)
#             if typestr is not None:
#                 return typestr
#         array = parameters.get("__array__")
#         if array is not None:
#             typestr = typestrs.get(array)
#             if typestr is not None:
#                 return typestr
#     return None


# def numba_record_typer(layouttype, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     rec = layouttype.parameters.get("__record__")
#     if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
#         typer = behavior["__numba_typer__", rec]
#         if callable(typer):
#             return typer
#     return None


# def numba_record_lower(layouttype, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     rec = layouttype.parameters.get("__record__")
#     if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
#         lower = behavior["__numba_lower__", rec]
#         if callable(lower):
#             return lower
#     return None


# def overload(behavior, signature):
#     if not any(s is None for s in signature):
#         behavior = Behavior(ak.behavior, behavior)
#         for key, custom in behavior.items():
#             if (
#                 isinstance(key, tuple)
#                 and len(key) == len(signature)
#                 and key[0] == signature[0]
#                 and all(
#                     k == s
#                     or (
#                         isinstance(k, type) and isinstance(s, type) and issubclass(s, k)
#                     )
#                     for k, s in zip(key[1:], signature[1:])
#                 )
#             ):
#                 return custom


# def numba_attrs(layouttype, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     rec = layouttype.parameters.get("__record__")
#     if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
#         for key, typer in behavior.items():
#             if (
#                 isinstance(key, tuple)
#                 and len(key) == 3
#                 and key[0] == "__numba_typer__"
#                 and key[1] == rec
#             ):
#                 lower = behavior["__numba_lower__", key[1], key[2]]
#                 yield key[2], typer, lower


# def numba_methods(layouttype, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     rec = layouttype.parameters.get("__record__")
#     if isinstance(rec, str) or (py27 and isinstance(rec, unicode)):
#         for key, typer in behavior.items():
#             if (
#                 isinstance(key, tuple)
#                 and len(key) == 4
#                 and key[0] == "__numba_typer__"
#                 and key[1] == rec
#                 and key[3] == ()
#             ):
#                 lower = behavior["__numba_lower__", key[1], key[2], ()]
#                 yield key[2], typer, lower


# def numba_unaryops(unaryop, left, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     done = False

#     if isinstance(left, ak._v2._connect.numba.layout.ContentType):
#         left = left.parameters.get("__record__")
#         if not (isinstance(left, str) or (py27 and isinstance(left, unicode))):
#             done = True

#     if not done:
#         for key, typer in behavior.items():
#             if (
#                 isinstance(key, tuple)
#                 and len(key) == 3
#                 and key[0] == "__numba_typer__"
#                 and key[1] == unaryop
#                 and key[2] == left
#             ):
#                 lower = behavior["__numba_lower__", key[1], key[2]]
#                 yield typer, lower


# def numba_binops(binop, left, right, behavior):
#     behavior = Behavior(ak.behavior, behavior)
#     done = False

#     if isinstance(left, ak._v2._connect.numba.layout.ContentType):
#         left = left.parameters.get("__record__")
#         if not (isinstance(left, str) or (py27 and isinstance(left, unicode))):
#             done = True

#     if isinstance(right, ak._v2._connect.numba.layout.ContentType):
#         right = right.parameters.get("__record__")
#         if not isinstance(right, str) and not (py27 and isinstance(right, unicode)):
#             done = True

#     if not done:
#         for key, typer in behavior.items():
#             if (
#                 isinstance(key, tuple)
#                 and len(key) == 4
#                 and key[0] == "__numba_typer__"
#                 and key[1] == left
#                 and key[2] == binop
#                 and key[3] == right
#             ):
#                 lower = behavior["__numba_lower__", key[1], key[2], key[3]]
#                 yield typer, lower


def behavior_of(*arrays, **kwargs):
    behavior = kwargs.get("behavior")
    if behavior is not None:
        # An explicit 'behavior' always wins.
        return behavior

    copied = False
    highs = (
        ak._v2.highlevel.Array,
        ak._v2.highlevel.Record,
        ak._v2.highlevel.ArrayBuilder,
    )
    for x in arrays[::-1]:
        if isinstance(x, highs) and x.behavior is not None:
            if behavior is None:
                behavior = x.behavior
            elif behavior is x.behavior:
                pass
            elif not copied:
                behavior = dict(behavior)
                behavior.update(x.behavior)
                copied = True
            else:
                behavior.update(x.behavior)
    return behavior


# maybe_wrap and maybe_wrap_like go here
def wrap(content, behavior=None, highlevel=True, like=None):
    if highlevel:
        if like is not None and behavior is None:
            behavior = behavior_of(like)

        if isinstance(content, ak._v2.contents.Content):
            return ak._v2.highlevel.Array(content, behavior=behavior)
        elif isinstance(content, ak._v2.record.Record):
            return ak._v2.highlevel.Record(content, behavior=behavior)

    return content


# def extra(args, kwargs, defaults):
#     out = []
#     for i in range(len(defaults)):
#         name, default = defaults[i]
#         if i < len(args):
#             out.append(args[i])
#         elif name in kwargs:
#             out.append(kwargs[name])
#         else:
#             out.append(default)
#     return out


# def key2index(keys, key):
#     if keys is None:
#         attempt = None
#     else:
#         try:
#             attempt = keys.index(key)
#         except ValueError:
#             attempt = None

#     if attempt is None:
#         m = key2index._pattern.match(key)
#         if m is not None:
#             attempt = m.group(0)

#     if attempt is None:
#         raise ValueError(
#             "key {0} not found in record".format(repr(key))
#         )
#     else:
#         return attempt


# key2index._pattern = re.compile(r"^[1-9][0-9]*$")


# def completely_flatten(array):
#     if isinstance(array, ak.partition.PartitionedArray):
#         out = []
#         for partition in array.partitions:
#             for outi in completely_flatten(partition):
#                 out.append(outi)
#         return tuple(out)

#     elif isinstance(array, virtualtypes):
#         return completely_flatten(array.array)

#     elif isinstance(array, unknowntypes):
#         return (ak.nplike.of(array).array([], dtype=np.bool_),)

#     elif isinstance(array, indexedtypes):
#         return completely_flatten(array.project())

#     elif isinstance(array, uniontypes):
#         out = []
#         for i in range(array.numcontents):
#             tmp = completely_flatten(array.project(i))
#             assert isinstance(tmp, tuple)
#             for x in tmp:
#                 out.append(x)
#         return tuple(out)

#     elif isinstance(array, optiontypes):
#         return completely_flatten(array.project())

#     elif isinstance(array, listtypes):
#         return completely_flatten(array.flatten(axis=1))

#     elif isinstance(array, recordtypes):
#         out = []
#         for i in range(array.numfields):
#             out.extend(completely_flatten(array.field(i)))
#         return tuple(out)

#     elif isinstance(array, ak._v2.contents.NumpyArray):
#         if array.format.upper().startswith("M"):
#             return (
#                 ak.nplike.of(array)
#                 .asarray(array.view_int64)
#                 .view(array.format)
#                 .reshape(-1),
#             )
#         else:
#             return (ak.nplike.of(array).asarray(array).reshape(-1),)

#     else:
#         raise RuntimeError(
#             "cannot completely flatten: {0}".format(type(array))
#
#         )


# def broadcast_and_apply(  # noqa: C901
#     inputs,
#     getfunction,
#     behavior,
#     allow_records=True,
#     pass_depth=True,
#     pass_user=False,
#     user=None,
#     left_broadcast=True,
#     right_broadcast=True,
#     numpy_to_regular=False,
#     regular_to_jagged=False,
# ):
#     def checklength(inputs):
#         length = len(inputs[0])
#         for x in inputs[1:]:
#             if len(x) != length:
#                 raise ValueError(
#                     "cannot broadcast {0} of length {1} with {2} of "
#                     "length {3}".format(
#                         type(inputs[0]).__name__, length, type(x).__name__, len(x)
#                     )
#
#                 )

#     def all_same_offsets(nplike, inputs):
#         offsets = None
#         for x in inputs:
#             if isinstance(
#                 x,
#                 (
#                     ak._v2.contents.ListOffsetArray32,
#                     ak._v2.contents.ListOffsetArrayU32,
#                     ak._v2.contents.ListOffsetArray64,
#                 ),
#             ):
#                 if offsets is None:
#                     offsets = nplike.asarray(x.offsets)
#                 elif not nplike.array_equal(offsets, nplike.asarray(x.offsets)):
#                     return False
#             elif isinstance(
#                 x,
#                 (
#                     ak._v2.contents.ListArray32,
#                     ak._v2.contents.ListArrayU32,
#                     ak._v2.contents.ListArray64,
#                 ),
#             ):
#                 starts = nplike.asarray(x.starts)
#                 stops = nplike.asarray(x.stops)
#                 if not nplike.array_equal(starts[1:], stops[:-1]):
#                     return False
#                 if offsets is None:
#                     offsets = nplike.empty(len(starts) + 1, dtype=starts.dtype)
#                     if len(offsets) == 1:
#                         offsets[0] = 0
#                     else:
#                         offsets[:-1] = starts
#                         offsets[-1] = stops[-1]
#                 elif not nplike.array_equal(offsets[:-1], starts) or (
#                     len(stops) != 0 and offsets[-1] != stops[-1]
#                 ):
#                     return False
#             elif isinstance(x, ak._v2.contents.RegularArray):
#                 if x.size == 0:
#                     my_offsets = nplike.empty(0, dtype=np.int64)
#                 else:
#                     my_offsets = nplike.arange(0, len(x.content), x.size)
#                 if offsets is None:
#                     offsets = my_offsets
#                 elif not nplike.array_equal(offsets, my_offsets):
#                     return False
#             elif isinstance(x, ak._v2.contents.Content):
#                 return False
#         else:
#             return True

#     def apply(inputs, depth, user):
#         nplike = ak.nplike.of(*inputs)

#         if numpy_to_regular:
#             inputs = [
#                 x.toRegularArray() if isinstance(x, ak._v2.contents.NumpyArray) else x
#                 for x in inputs
#             ]

#         if regular_to_jagged:
#             inputs = [
#                 x.toListOffsetArray64(False)
#                 if isinstance(x, ak._v2.contents.RegularArray)
#                 else x
#                 for x in inputs
#             ]

#         # handle implicit right-broadcasting (i.e. NumPy-like)
#         if right_broadcast and any(isinstance(x, listtypes) for x in inputs):
#             maxdepth = max(
#                 x.purelist_depth for x in inputs if isinstance(x, ak._v2.contents.Content)
#             )

#             if maxdepth > 0 and all(
#                 x.purelist_isregular for x in inputs if isinstance(x, ak._v2.contents.Content)
#             ):
#                 nextinputs = []
#                 for obj in inputs:
#                     if isinstance(obj, ak._v2.contents.Content):
#                         while obj.purelist_depth < maxdepth:
#                             obj = ak._v2.contents.RegularArray(obj, 1, len(obj))
#                     nextinputs.append(obj)
#                 if any(x is not y for x, y in zip(inputs, nextinputs)):
#                     return apply(nextinputs, depth, user)

#         # now all lengths must agree
#         checklength([x for x in inputs if isinstance(x, ak._v2.contents.Content)])

#         args = ()
#         if pass_depth:
#             args = args + (depth,)
#         if pass_user:
#             args = args + (user,)

#         custom = getfunction(inputs, *args)
#         if callable(custom):
#             return custom()
#         else:
#             user = custom

#         # the rest of this is one switch statement
#         if any(isinstance(x, virtualtypes) for x in inputs):
#             nextinputs = []
#             for x in inputs:
#                 if isinstance(x, virtualtypes):
#                     nextinputs.append(x.array)
#                 else:
#                     nextinputs.append(x)
#             return apply(nextinputs, depth, user)

#         elif any(isinstance(x, unknowntypes) for x in inputs):
#             nextinputs = []
#             for x in inputs:
#                 if isinstance(x, unknowntypes):
#                     nextinputs.append(
#                         ak._v2.contents.NumpyArray(nplike.array([], dtype=np.bool_))
#                     )
#                 else:
#                     nextinputs.append(x)
#             return apply(nextinputs, depth, user)

#         elif any(isinstance(x, ak._v2.contents.NumpyArray) and x.ndim > 1 for x in inputs):
#             nextinputs = []
#             for x in inputs:
#                 if isinstance(x, ak._v2.contents.NumpyArray) and x.ndim > 1:
#                     nextinputs.append(x.toRegularArray())
#                 else:
#                     nextinputs.append(x)
#             return apply(nextinputs, depth, user)

#         elif any(isinstance(x, indexedtypes) for x in inputs):
#             nextinputs = []
#             for x in inputs:
#                 if isinstance(x, indexedtypes):
#                     nextinputs.append(x.project())
#                 else:
#                     nextinputs.append(x)
#             return apply(nextinputs, depth, user)

#         elif any(isinstance(x, uniontypes) for x in inputs):
#             tagslist = []
#             numtags = []
#             length = None
#             for x in inputs:
#                 if isinstance(x, uniontypes):
#                     tagslist.append(nplike.asarray(x.tags))
#                     numtags.append(len(x.contents))
#                     if length is None:
#                         length = len(tagslist[-1])
#                     elif length != len(tagslist[-1]):
#                         raise ValueError(
#                             "cannot broadcast UnionArray of length {0} "
#                             "with UnionArray of length {1}".format(
#                                 length, len(tagslist[-1])
#                             )
#
#                         )

#             combos = nplike.stack(tagslist, axis=-1)

#             all_combos = nplike.array(
#                 list(itertools.product(*[range(x) for x in numtags])),
#                 dtype=[(str(i), combos.dtype) for i in range(len(tagslist))],
#             )

#             combos = combos.view(
#                 [(str(i), combos.dtype) for i in range(len(tagslist))]
#             ).reshape(length)

#             tags = nplike.empty(length, dtype=np.int8)
#             index = nplike.empty(length, dtype=np.int64)
#             numoutputs = None
#             outcontents = []
#             for tag, combo in enumerate(all_combos):
#                 mask = combos == combo
#                 tags[mask] = tag
#                 index[mask] = nplike.arange(nplike.count_nonzero(mask))
#                 nextinputs = []
#                 i = 0
#                 for x in inputs:
#                     if isinstance(x, uniontypes):
#                         nextinputs.append(x[mask].project(combo[str(i)]))
#                         i += 1
#                     elif isinstance(x, ak._v2.contents.Content):
#                         nextinputs.append(x[mask])
#                     else:
#                         nextinputs.append(x)
#                 outcontents.append(apply(nextinputs, depth, user))
#                 assert isinstance(outcontents[-1], tuple)
#                 if numoutputs is not None:
#                     assert numoutputs == len(outcontents[-1])
#                 numoutputs = len(outcontents[-1])

#             assert numoutputs is not None

#             tags = ak._v2.contents.Index8(tags)
#             index = ak._v2.contents.Index64(index)
#             return tuple(
#                 ak._v2.contents.UnionArray8_64(
#                     tags, index, [x[i] for x in outcontents]
#                 ).simplify()
#                 for i in range(numoutputs)
#             )

#         elif any(isinstance(x, optiontypes) for x in inputs):
#             mask = None
#             for x in inputs:
#                 if isinstance(x, optiontypes):
#                     m = nplike.asarray(x.bytemask()).view(np.bool_)
#                     if mask is None:
#                         mask = m
#                     else:
#                         nplike.bitwise_or(mask, m, out=mask)

#             nextmask = ak._v2.contents.Index8(mask.view(np.int8))
#             index = nplike.full(len(mask), -1, dtype=np.int64)
#             index[~mask] = nplike.arange(
#                 len(mask) - nplike.count_nonzero(mask), dtype=np.int64
#             )
#             index = ak._v2.contents.Index64(index)
#             if any(not isinstance(x, optiontypes) for x in inputs):
#                 nextindex = nplike.arange(len(mask), dtype=np.int64)
#                 nextindex[mask] = -1
#                 nextindex = ak._v2.contents.Index64(nextindex)

#             nextinputs = []
#             for x in inputs:
#                 if isinstance(x, optiontypes):
#                     nextinputs.append(x.project(nextmask))
#                 elif isinstance(x, ak._v2.contents.Content):
#                     nextinputs.append(
#                         ak._v2.contents.IndexedOptionArray64(nextindex, x).project(nextmask)
#                     )
#                 else:
#                     nextinputs.append(x)

#             outcontent = apply(nextinputs, depth, user)
#             assert isinstance(outcontent, tuple)
#             return tuple(
#                 ak._v2.contents.IndexedOptionArray64(index, x).simplify() for x in outcontent
#             )

#         elif any(isinstance(x, listtypes) for x in inputs):
#             if all(
#                 isinstance(x, ak._v2.contents.RegularArray) or not isinstance(x, listtypes)
#                 for x in inputs
#             ):
#                 maxsize = max(
#                     [x.size for x in inputs if isinstance(x, ak._v2.contents.RegularArray)]
#                 )
#                 for x in inputs:
#                     if isinstance(x, ak._v2.contents.RegularArray):
#                         if maxsize > 1 and x.size == 1:
#                             tmpindex = ak._v2.contents.Index64(
#                                 nplike.repeat(
#                                     nplike.arange(len(x), dtype=np.int64), maxsize
#                                 )
#                             )
#                 nextinputs = []
#                 for x in inputs:
#                     if isinstance(x, ak._v2.contents.RegularArray):
#                         if maxsize > 1 and x.size == 1:
#                             nextinputs.append(
#                                 ak._v2.contents.IndexedArray64(
#                                     tmpindex, x.content[: len(x) * x.size]
#                                 ).project()
#                             )
#                         elif x.size == maxsize:
#                             nextinputs.append(x.content[: len(x) * x.size])
#                         else:
#                             raise ValueError(
#                                 "cannot broadcast RegularArray of size "
#                                 "{0} with RegularArray of size {1}".format(
#                                     x.size, maxsize
#                                 )
#
#                             )
#                     else:
#                         nextinputs.append(x)

#                 maxlen = max(
#                     [len(x) for x in nextinputs if isinstance(x, ak._v2.contents.Content)]
#                 )
#                 outcontent = apply(nextinputs, depth + 1, user)
#                 assert isinstance(outcontent, tuple)

#                 return tuple(
#                     ak._v2.contents.RegularArray(x, maxsize, maxlen) for x in outcontent
#                 )

#             elif not all_same_offsets(nplike, inputs):
#                 fcns = [
#                     custom_broadcast(x, behavior)
#                     if isinstance(x, ak._v2.contents.Content)
#                     else None
#                     for x in inputs
#                 ]

#                 first, secondround = None, False
#                 for x, fcn in zip(inputs, fcns):
#                     if (
#                         isinstance(x, listtypes)
#                         and not isinstance(x, ak._v2.contents.RegularArray)
#                         and fcn is None
#                     ):
#                         first = x
#                         break

#                 if first is None:
#                     secondround = True
#                     for x in inputs:
#                         if isinstance(x, listtypes) and not isinstance(
#                             x, ak._v2.contents.RegularArray
#                         ):
#                             first = x
#                             break

#                 offsets = first.compact_offsets64(True)

#                 nextinputs = []
#                 for x, fcn in zip(inputs, fcns):
#                     if callable(fcn) and not secondround:
#                         nextinputs.append(fcn(x, offsets))
#                     elif isinstance(x, listtypes):
#                         nextinputs.append(x.broadcast_tooffsets64(offsets).content)
#                     # handle implicit left-broadcasting (unlike NumPy)
#                     elif left_broadcast and isinstance(x, ak._v2.contents.Content):
#                         nextinputs.append(
#                             ak._v2.contents.RegularArray(x, 1, len(x))
#                             .broadcast_tooffsets64(offsets)
#                             .content
#                         )
#                     else:
#                         nextinputs.append(x)

#                 outcontent = apply(nextinputs, depth + 1, user)
#                 assert isinstance(outcontent, tuple)

#                 return tuple(
#                     ak._v2.contents.ListOffsetArray64(offsets, x) for x in outcontent
#                 )

#             else:
#                 lencontent, offsets, starts, stops = None, None, None, None
#                 nextinputs = []

#                 for x in inputs:
#                     if isinstance(
#                         x,
#                         (
#                             ak._v2.contents.ListOffsetArray32,
#                             ak._v2.contents.ListOffsetArrayU32,
#                             ak._v2.contents.ListOffsetArray64,
#                         ),
#                     ):
#                         offsets = x.offsets
#                         lencontent = offsets[-1]
#                         nextinputs.append(x.content[:lencontent])

#                     elif isinstance(
#                         x,
#                         (
#                             ak._v2.contents.ListArray32,
#                             ak._v2.contents.ListArrayU32,
#                             ak._v2.contents.ListArray64,
#                         ),
#                     ):
#                         starts, stops = x.starts, x.stops
#                         if len(starts) == 0 or len(stops) == 0:
#                             nextinputs.append(x.content[:0])
#                         else:
#                             lencontent = nplike.max(stops)
#                             nextinputs.append(x.content[:lencontent])

#                     else:
#                         nextinputs.append(x)

#                 outcontent = apply(nextinputs, depth + 1, user)

#                 if isinstance(offsets, ak._v2.contents.Index32):
#                     return tuple(
#                         ak._v2.contents.ListOffsetArray32(offsets, x) for x in outcontent
#                     )
#                 elif isinstance(offsets, ak._v2.contents.IndexU32):
#                     return tuple(
#                         ak._v2.contents.ListOffsetArrayU32(offsets, x) for x in outcontent
#                     )
#                 elif isinstance(offsets, ak._v2.contents.Index64):
#                     return tuple(
#                         ak._v2.contents.ListOffsetArray64(offsets, x) for x in outcontent
#                     )
#                 elif isinstance(starts, ak._v2.contents.Index32):
#                     return tuple(
#                         ak._v2.contents.ListArray32(starts, stops, x) for x in outcontent
#                     )
#                 elif isinstance(starts, ak._v2.contents.IndexU32):
#                     return tuple(
#                         ak._v2.contents.ListArrayU32(starts, stops, x) for x in outcontent
#                     )
#                 elif isinstance(starts, ak._v2.contents.Index64):
#                     return tuple(
#                         ak._v2.contents.ListArray64(starts, stops, x) for x in outcontent
#                     )
#                 else:
#                     raise AssertionError(
#                         "unexpected offsets, starts: {0} {1}".format(
#                             type(offsets), type(starts)
#                         )
#
#                     )

#         elif any(isinstance(x, recordtypes) for x in inputs):
#             if not allow_records:
#                 raise ValueError(
#                     "cannot broadcast records in this type of operation"
#
#                 )

#             keys = None
#             length = None
#             istuple = True
#             for x in inputs:
#                 if isinstance(x, recordtypes):
#                     if keys is None:
#                         keys = x.keys()
#                     elif set(keys) != set(x.keys()):
#                         raise ValueError(
#                             "cannot broadcast records because keys don't "
#                             "match:\n    {0}\n    {1}".format(
#                                 ", ".join(sorted(keys)), ", ".join(sorted(x.keys()))
#                             )
#
#                         )
#                     if length is None:
#                         length = len(x)
#                     elif length != len(x):
#                         raise ValueError(
#                             "cannot broadcast RecordArray of length {0} "
#                             "with RecordArray of length {1}".format(length, len(x))
#
#                         )
#                     if not x.istuple:
#                         istuple = False

#             outcontents = []
#             numoutputs = None
#             for key in keys:
#                 outcontents.append(
#                     apply(
#                         [
#                             x if not isinstance(x, recordtypes) else x[key]
#                             for x in inputs
#                         ],
#                         depth,
#                         user,
#                     )
#                 )
#                 assert isinstance(outcontents[-1], tuple)
#                 if numoutputs is not None:
#                     assert numoutputs == len(outcontents[-1])
#                 numoutputs = len(outcontents[-1])
#             return tuple(
#                 ak._v2.contents.RecordArray(
#                     [x[i] for x in outcontents], None if istuple else keys, length
#                 )
#                 for i in range(numoutputs)
#             )

#         else:
#             raise ValueError(
#                 "cannot broadcast: {0}".format(", ".join(repr(type(x)) for x in inputs))
#
#             )

#     if any(isinstance(x, ak.partition.PartitionedArray) for x in inputs):
#         purelist_isregular = True
#         purelist_depths = set()
#         for x in inputs:
#             if isinstance(x, (ak._v2.contents.Content, ak.partition.PartitionedArray)):
#                 if not x.purelist_isregular:
#                     purelist_isregular = False
#                     break
#                 purelist_depths.add(x.purelist_depth)

#         if purelist_isregular and len(purelist_depths) > 1:
#             nextinputs = []
#             for x in inputs:
#                 if isinstance(x, ak.partition.PartitionedArray):
#                     nextinputs.append(x.toContent())
#                 else:
#                     nextinputs.append(x)

#             isscalar = []
#             out = apply(broadcast_pack(nextinputs, isscalar), 0, None)
#             assert isinstance(out, tuple)
#             return tuple(broadcast_unpack(x, isscalar) for x in out)

#         else:
#             sample = None
#             for x in inputs:
#                 if isinstance(x, ak.partition.PartitionedArray):
#                     sample = x
#                     break
#             nextinputs = ak.partition.partition_as(sample, inputs)

#             outputs = []
#             for part_inputs in ak.partition.iterate(sample.numpartitions, nextinputs):
#                 isscalar = []
#                 part = apply(broadcast_pack(part_inputs, isscalar), 0, None)
#                 assert isinstance(part, tuple)
#                 outputs.append(tuple(broadcast_unpack(x, isscalar) for x in part))

#             out = ()
#             for i in range(len(part)):
#                 out = out + (
#                     ak.partition.IrregularlyPartitionedArray([x[i] for x in outputs]),
#                 )
#             return out

#     else:
#         isscalar = []
#         out = apply(broadcast_pack(inputs, isscalar), 0, user)
#         assert isinstance(out, tuple)
#         return tuple(broadcast_unpack(x, isscalar) for x in out)


# def broadcast_pack(inputs, isscalar):
#     maxlen = -1
#     for x in inputs:
#         if isinstance(x, ak._v2.contents.Content):
#             maxlen = max(maxlen, len(x))
#     if maxlen < 0:
#         maxlen = 1

#     nextinputs = []
#     for x in inputs:
#         if isinstance(x, ak._v2.record.Record):
#             index = ak.nplike.of(*inputs).full(maxlen, x.at, dtype=np.int64)
#             nextinputs.append(ak._v2.contents.RegularArray(x.array[index], maxlen, 1))
#             isscalar.append(True)
#         elif isinstance(x, ak._v2.contents.Content):
#             nextinputs.append(ak._v2.contents.RegularArray(x, len(x), 1))
#             isscalar.append(False)
#         else:
#             nextinputs.append(x)
#             isscalar.append(True)

#     return nextinputs


# def broadcast_unpack(x, isscalar):
#     if all(isscalar):
#         if len(x) == 0:
#             return x.getitem_nothing().getitem_nothing()
#         else:
#             return x[0][0]
#     else:
#         if len(x) == 0:
#             return x.getitem_nothing()
#         else:
#             return x[0]


# def recursively_apply(
#     layout,
#     getfunction,
#     pass_depth=True,
#     pass_user=False,
#     user=None,
#     keep_parameters=True,
#     numpy_to_regular=False,
# ):
#     def transform(layout, depth, user):
#         if numpy_to_regular and isinstance(layout, ak._v2.contents.NumpyArray):
#             layout = layout.toRegularArray()

#         args = ()
#         if pass_depth:
#             args = args + (depth,)
#         if pass_user:
#             args = args + (user,)

#         custom = getfunction(layout, *args)
#         if callable(custom):
#             return custom()

#         else:
#             return transform_child_layouts(
#                 transform, layout, depth, user=custom, keep_parameters=keep_parameters
#             )

#     return transform(layout, 1, user)


# def recursive_walk(layout, apply, args=(), depth=1, materialize=False):
#     apply(layout, depth, *args)

#     if isinstance(layout, ak.partition.PartitionedArray):
#         for x in layout.partitions:
#             recursive_walk(x, apply, args, depth, materialize)

#     elif isinstance(layout, ak._v2.contents.NumpyArray):
#         pass

#     elif isinstance(layout, ak._v2.contents.EmptyArray):
#         pass

#     elif isinstance(
#         layout,
#         (
#             ak._v2.contents.RegularArray,
#             ak._v2.contents.ListArray32,
#             ak._v2.contents.ListArrayU32,
#             ak._v2.contents.ListArray64,
#             ak._v2.contents.ListOffsetArray32,
#             ak._v2.contents.ListOffsetArrayU32,
#             ak._v2.contents.ListOffsetArray64,
#         ),
#     ):
#         recursive_walk(layout.content, apply, args, depth + 1, materialize)

#     elif isinstance(
#         layout,
#         (
#             ak._v2.contents.IndexedArray32,
#             ak._v2.contents.IndexedArrayU32,
#             ak._v2.contents.IndexedArray64,
#             ak._v2.contents.IndexedOptionArray32,
#             ak._v2.contents.IndexedOptionArray64,
#             ak._v2.contents.ByteMaskedArray,
#             ak._v2.contents.BitMaskedArray,
#             ak._v2.contents.UnmaskedArray,
#         ),
#     ):
#         recursive_walk(layout.content, apply, args, depth, materialize)

#     elif isinstance(
#         layout,
#         (
#             ak._v2.contents.RecordArray,
#             ak._v2.contents.UnionArray8_32,
#             ak._v2.contents.UnionArray8_U32,
#             ak._v2.contents.UnionArray8_64,
#         ),
#     ):
#         for x in layout.contents:
#             recursive_walk(x, apply, args, depth, materialize)

#     elif isinstance(layout, ak._v2.record.Record):
#         recursive_walk(layout.array, apply, args, depth, materialize)

#     elif isinstance(layout, ak._v2.contents.VirtualArray):
#         if materialize:
#             recursive_walk(layout.array, apply, args, depth, materialize)

#     else:
#         raise AssertionError(
#             "unrecognized Content type: {0}".format(type(layout))
#
#         )


# def transform_child_layouts(transform, layout, depth, user=None, keep_parameters=True):
#     # the rest of this is one switch statement
#     if isinstance(layout, ak.partition.PartitionedArray):
#         return ak.partition.IrregularlyPartitionedArray(
#             [transform(x, depth, user) for x in layout.partitions]
#         )

#     elif isinstance(layout, ak._v2.contents.NumpyArray):
#         if keep_parameters:
#             return layout
#         else:
#             return ak._v2.contents.NumpyArray(
#                 ak.nplike.of(layout).asarray(layout), layout.identities, None
#             )

#     elif isinstance(layout, ak._v2.contents.EmptyArray):
#         if keep_parameters:
#             return layout
#         else:
#             return ak._v2.contents.EmptyArray(layout.identities, None)

#     elif isinstance(layout, ak._v2.contents.RegularArray):
#         return ak._v2.contents.RegularArray(
#             transform(layout.content, depth + 1, user),
#             layout.size,
#             len(layout),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.ListArray32):
#         return ak._v2.contents.ListArray32(
#             layout.starts,
#             layout.stops,
#             transform(layout.content, depth + 1, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.ListArrayU32):
#         return ak._v2.contents.ListArrayU32(
#             layout.starts,
#             layout.stops,
#             transform(layout.content, depth + 1, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.ListArray64):
#         return ak._v2.contents.ListArray64(
#             layout.starts,
#             layout.stops,
#             transform(layout.content, depth + 1, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.ListOffsetArray32):
#         return ak._v2.contents.ListOffsetArray32(
#             layout.offsets,
#             transform(layout.content, depth + 1, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.ListOffsetArrayU32):
#         return ak._v2.contents.ListOffsetArrayU32(
#             layout.offsets,
#             transform(layout.content, depth + 1, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.ListOffsetArray64):
#         return ak._v2.contents.ListOffsetArray64(
#             layout.offsets,
#             transform(layout.content, depth + 1, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.IndexedArray32):
#         return ak._v2.contents.IndexedArray32(
#             layout.index,
#             transform(layout.content, depth, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.IndexedArrayU32):
#         return ak._v2.contents.IndexedArrayU32(
#             layout.index,
#             transform(layout.content, depth, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.IndexedArray64):
#         return ak._v2.contents.IndexedArray64(
#             layout.index,
#             transform(layout.content, depth, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.IndexedOptionArray32):
#         return ak._v2.contents.IndexedOptionArray32(
#             layout.index,
#             transform(layout.content, depth, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.IndexedOptionArray64):
#         return ak._v2.contents.IndexedOptionArray64(
#             layout.index,
#             transform(layout.content, depth, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.ByteMaskedArray):
#         return ak._v2.contents.ByteMaskedArray(
#             layout.mask,
#             transform(layout.content, depth, user),
#             layout.valid_when,
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.BitMaskedArray):
#         return ak._v2.contents.BitMaskedArray(
#             layout.mask,
#             transform(layout.content, depth, user),
#             layout.valid_when,
#             len(layout),
#             layout.lsb_order,
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.UnmaskedArray):
#         return ak._v2.contents.UnmaskedArray(
#             transform(layout.content, depth, user),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.RecordArray):
#         return ak._v2.contents.RecordArray(
#             [transform(x, depth, user) for x in layout.contents],
#             layout.recordlookup,
#             len(layout),
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.record.Record):
#         return ak._v2.record.Record(
#             transform(layout.array, depth, user),
#             layout.at,
#         )

#     elif isinstance(layout, ak._v2.contents.UnionArray8_32):
#         return ak._v2.contents.UnionArray8_32(
#             layout.tags,
#             layout.index,
#             [transform(x, depth, user) for x in layout.contents],
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.UnionArray8_U32):
#         return ak._v2.contents.UnionArray8_U32(
#             layout.tags,
#             layout.index,
#             [transform(x, depth, user) for x in layout.contents],
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.UnionArray8_64):
#         return ak._v2.contents.UnionArray8_64(
#             layout.tags,
#             layout.index,
#             [transform(x, depth, user) for x in layout.contents],
#             layout.identities,
#             layout.parameters if keep_parameters else None,
#         )

#     elif isinstance(layout, ak._v2.contents.VirtualArray):
#         return transform(layout.array, depth, user)

#     else:
#         raise AssertionError(
#             "unrecognized Content type: {0}".format(type(layout))
#
#         )


# def highlevel_type(layout, behavior, isarray):
#     if isarray:
#         return ak.types.ArrayType(layout.type(typestrs(behavior)), len(layout))
#     else:
#         return layout.type(typestrs(behavior))


# _is_identifier = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


# def minimally_touching_string(limit_length, layout, behavior):
#     if isinstance(layout, ak._v2.layout.Record):
#         layout = layout.array[layout.at : layout.at + 1]

#     if len(layout) == 0:
#         return "[]"

#     def forward(x, space, brackets=True, wrap=True, stop=None):
#         done = False
#         if wrap and isinstance(x, ak._v2.contents.Content):
#             cls = arrayclass(x, behavior)
#             if cls is not ak._v2.highlevel.Array:
#                 y = cls(x, behavior=behavior)
#                 if "__repr__" in type(y).__dict__:
#                     yield space + repr(y)
#                     done = True
#         if wrap and isinstance(x, ak._v2.record.Record):
#             cls = recordclass(x, behavior)
#             if cls is not ak._v2.highlevel.Record:
#                 y = cls(x, behavior=behavior)
#                 if "__repr__" in type(y).__dict__:
#                     yield space + repr(y)
#                     done = True
#         if not done:
#             if isinstance(x, ak._v2.contents.Content):
#                 if brackets:
#                     yield space + "["
#                 sp = ""
#                 for i in range(len(x) if stop is None else stop):
#                     for token in forward(x[i], sp):
#                         yield token
#                     sp = ", "
#                 if brackets:
#                     yield "]"
#             elif isinstance(x, ak._v2.record.Record) and x.istuple:
#                 yield space + "("
#                 sp = ""
#                 for i in range(x.numfields):
#                     key = sp
#                     for token in forward(x[str(i)], ""):
#                         yield key + token
#                         key = ""
#                     sp = ", "
#                 yield ")"
#             elif isinstance(x, ak._v2.record.Record):
#                 yield space + "{"
#                 sp = ""
#                 for k in x.keys():
#                     if _is_identifier.match(k) is None:
#                         kk = repr(k)
#                         if kk.startswith("u"):
#                             kk = kk[1:]
#                     else:
#                         kk = k
#                     key = sp + kk + ": "
#                     for token in forward(x[k], ""):
#                         yield key + token
#                         key = ""
#                     sp = ", "
#                 yield "}"
#             elif isinstance(x, (np.datetime64, np.timedelta64)):
#                 yield space + str(x)
#             elif isinstance(x, (float, np.floating)):
#                 yield space + "{0:.3g}".format(x)
#             else:
#                 yield space + repr(x)

#     def backward(x, space, brackets=True, wrap=True, stop=-1):
#         done = False
#         if wrap and isinstance(x, ak._v2.contents.Content):
#             cls = arrayclass(x, behavior)
#             if cls is not ak._v2.highlevel.Array:
#                 y = cls(x, behavior=behavior)
#                 if "__repr__" in type(y).__dict__:
#                     yield repr(y) + space
#                     done = True
#         if wrap and isinstance(x, ak._v2.record.Record):
#             cls = recordclass(x, behavior)
#             if cls is not ak._v2.highlevel.Record:
#                 y = cls(x, behavior=behavior)
#                 if "__repr__" in type(y).__dict__:
#                     yield repr(y) + space
#                     done = True
#         if not done:
#             if isinstance(x, ak._v2.contents.Content):
#                 if brackets:
#                     yield "]" + space
#                 sp = ""
#                 for i in range(len(x) - 1, stop, -1):
#                     for token in backward(x[i], sp):
#                         yield token
#                     sp = ", "
#                 if brackets:
#                     yield "["
#             elif isinstance(x, ak._v2.record.Record) and x.istuple:
#                 yield ")" + space
#                 for i in range(x.numfields - 1, -1, -1):
#                     last = None
#                     for token in backward(x[str(i)], ""):
#                         if last is not None:
#                             yield last
#                         last = token
#                     if last is not None:
#                         yield last
#                     if i != 0:
#                         yield ", "
#                 yield "("
#             elif isinstance(x, ak._v2.record.Record):
#                 yield "}" + space
#                 keys = x.keys()
#                 for i in range(len(keys) - 1, -1, -1):
#                     last = None
#                     for token in backward(x[keys[i]], ""):
#                         if last is not None:
#                             yield last
#                         last = token
#                     if _is_identifier.match(keys[i]) is None:
#                         kk = repr(keys[i])
#                         if kk.startswith("u"):
#                             kk = kk[1:]
#                     else:
#                         kk = keys[i]
#                     if last is not None:
#                         yield kk + ": " + last
#                     if i != 0:
#                         yield ", "
#                 yield "{"
#             elif isinstance(x, (np.datetime64, np.timedelta64)):
#                 yield str(x) + space
#             elif isinstance(x, (float, np.floating)):
#                 yield "{0:.3g}".format(x) + space
#             else:
#                 yield repr(x) + space

#     def forever(iterable):
#         for token in iterable:
#             yield token
#         while True:
#             yield None

#     halfway = len(layout) // 2
#     left, right = ["["], ["]"]
#     leftlen, rightlen = 1, 1
#     leftgen = forever(forward(layout, "", brackets=False, wrap=False, stop=halfway))
#     rightgen = forever(
#         backward(layout, "", brackets=False, wrap=False, stop=halfway - 1)
#     )
#     while True:
#         lft = next(leftgen)
#         rgt = next(rightgen)

#         if lft is not None:
#             if (
#                 leftlen
#                 + rightlen
#                 + len(lft)
#                 + (2 if lft is None and rgt is None else 6)
#                 > limit_length
#             ):
#                 break
#             left.append(lft)
#             leftlen += len(lft)

#         if rgt is not None:
#             if (
#                 leftlen
#                 + rightlen
#                 + len(rgt)
#                 + (2 if lft is None and rgt is None else 6)
#                 > limit_length
#             ):
#                 break
#             right.append(rgt)
#             rightlen += len(rgt)

#         if lft is None and rgt is None:
#             break

#     while len(left) > 1 and (
#         left[-1] == "["
#         or left[-1] == ", ["
#         or left[-1] == "{"
#         or left[-1] == ", {"
#         or left[-1] == ", "
#     ):
#         left.pop()
#         lft = ""
#     while len(right) > 1 and (
#         right[-1] == "]"
#         or right[-1] == "], "
#         or right[-1] == "}"
#         or right[-1] == "}, "
#         or right[-1] == ", "
#     ):
#         right.pop()
#         rgt = ""
#     if lft is None and rgt is None:
#         if left == ["["]:
#             return "[" + "".join(reversed(right)).lstrip(" ")
#         else:
#             return (
#                 "".join(left).rstrip(" ") + ", " + "".join(reversed(right)).lstrip(" ")
#             )
#     else:
#         if left == ["["] and right == ["]"]:
#             return "[...]"
#         elif left == ["["]:
#             return "[... " + "".join(reversed(right)).lstrip(" ")
#         else:
#             return (
#                 "".join(left).rstrip(" ")
#                 + ", ... "
#                 + "".join(reversed(right)).lstrip(" ")
#             )


# def make_union(tags, index, contents, identities, parameters):
#     if isinstance(index, ak._v2.contents.Index32):
#         return ak._v2.contents.UnionArray8_32(tags, index, contents, identities, parameters)
#     elif isinstance(index, ak._v2.contents.IndexU32):
#         return ak._v2.contents.UnionArray8_U32(tags, index, contents, identities, parameters)
#     elif isinstance(index, ak._v2.contents.Index64):
#         return ak._v2.contents.UnionArray8_64(tags, index, contents, identities, parameters)
#     else:
#         raise AssertionError(index)


# def union_to_record(unionarray, anonymous):
#     nplike = ak.nplike.of(unionarray)

#     contents = []
#     for layout in unionarray.contents:
#         if isinstance(layout, virtualtypes):
#             contents.append(layout.array)
#         elif isinstance(layout, indexedtypes):
#             contents.append(layout.project())
#         elif isinstance(layout, uniontypes):
#             contents.append(union_to_record(layout, anonymous))
#         elif isinstance(layout, optiontypes):
#             contents.append(
#                 ak.operations.structure.fill_none(
#                     layout, np.nan, axis=0, highlevel=False
#                 )
#             )
#         else:
#             contents.append(layout)

#     if not any(isinstance(x, ak._v2.contents.RecordArray) for x in contents):
#         return make_union(
#             unionarray.tags,
#             unionarray.index,
#             contents,
#             unionarray.identities,
#             unionarray.parameters,
#         )

#     else:
#         seen = set()
#         all_names = []
#         for layout in contents:
#             if isinstance(layout, ak._v2.contents.RecordArray):
#                 for key in layout.keys():
#                     if key not in seen:
#                         seen.add(key)
#                         all_names.append(key)
#             else:
#                 if anonymous not in seen:
#                     seen.add(anonymous)
#                     all_names.append(anonymous)

#         missingarray = ak._v2.contents.IndexedOptionArray64(
#             ak._v2.contents.Index64(nplike.full(len(unionarray), -1, dtype=np.int64)),
#             ak._v2.contents.EmptyArray(),
#         )

#         all_fields = []
#         for name in all_names:
#             union_contents = []
#             for layout in contents:
#                 if isinstance(layout, ak._v2.contents.RecordArray):
#                     for key in layout.keys():
#                         if name == key:
#                             union_contents.append(layout.field(key))
#                             break
#                     else:
#                         union_contents.append(missingarray)
#                 else:
#                     if name == anonymous:
#                         union_contents.append(layout)
#                     else:
#                         union_contents.append(missingarray)

#             all_fields.append(
#                 make_union(
#                     unionarray.tags,
#                     unionarray.index,
#                     union_contents,
#                     unionarray.identities,
#                     unionarray.parameters,
#                 ).simplify()
#             )

#         return ak._v2.contents.RecordArray(all_fields, all_names, len(unionarray))


# def adjust_old_pickle(form, container, num_partitions, behavior):
#     def key_format(**v):
#         if num_partitions is None:
#             if v["attribute"] == "data":
#                 return "{form_key}".format(**v)
#             else:
#                 return "{form_key}-{attribute}".format(**v)

#         else:
#             if v["attribute"] == "data":
#                 return "{form_key}-part{partition}".format(**v)
#             else:
#                 return "{form_key}-{attribute}-part{partition}".format(**v)

#     return ak.operations.convert.from_buffers(
#         form,
#         None,
#         container,
#         partition_start=0,
#         key_format=key_format,
#         lazy=False,
#         lazy_cache="new",
#         lazy_cache_key=None,
#         highlevel=False,
#         behavior=behavior,
#     )
