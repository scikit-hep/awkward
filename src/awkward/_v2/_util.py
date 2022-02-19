# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# First, transition all the _v2 code to start using implementations in this file.
# Then build up the high-level replacements.

# import re
# import os.path
# import warnings
import setuptools
import os
import numbers

from collections.abc import Mapping

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

_backends = {
    "cpu": ak.nplike.Numpy,
    "cuda": ak.nplike.Cupy,
}


def regularize_backend(backend):
    if backend in _backends:
        return _backends[backend].instance()
    else:
        raise ValueError("The available backends for now are `cpu` and `cuda`.")


def parse_version(version):
    return setuptools.extern.packaging.version.parse(version)


def numpy_at_least(version):
    import numpy

    return parse_version(numpy.__version__) >= parse_version(version)


def in_module(obj, modulename):
    m = type(obj).__module__
    return m == modulename or m.startswith(modulename + ".")


def is_file_path(x):
    try:
        return os.path.isfile(x)
    except ValueError:
        return False


def isint(x):
    return isinstance(x, (int, numbers.Integral, np.integer)) and not isinstance(
        x, (bool, np.bool_)
    )


def isnum(x):
    return isinstance(x, (int, float, numbers.Real, np.number)) and not isinstance(
        x, (bool, np.bool_)
    )


def isstr(x):
    return isinstance(x, str)


def tobytes(array):
    if hasattr(array, "tobytes"):
        return array.tobytes()
    else:
        return array.tostring()


def little_endian(array):
    return array.astype(array.dtype.newbyteorder("<"), copy=False)


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
# class Unspecified(object):
#     pass


def regularize_path(path):
    """
    Converts pathlib Paths into plain string paths (for all versions of Python).
    """
    is_path = False

    if isinstance(path, getattr(os, "PathLike", ())):
        is_path = True
        path = os.fspath(path)

    elif hasattr(path, "__fspath__"):
        is_path = True
        path = path.__fspath__()

    elif path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(path, pathlib.Path):
            is_path = True
            path = str(path)

    return is_path, path


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
        yield from self.keys()

    def __len__(self):
        return len(set(self.defaults) | set(self.overrides))


def arrayclass(layout, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
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


def custom_cast(obj, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    for key, fcn in behavior.items():
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "__cast__"
            and isinstance(obj, key[1])
        ):
            return fcn
    return None


def custom_broadcast(layout, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    custom = layout.parameter("__array__")
    if not isstr(custom):
        custom = layout.parameter("__record__")
    if not isstr(custom):
        custom = layout.purelist_parameter("__record__")
    if isstr(custom):
        for key, fcn in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] == "__broadcast__"
                and key[1] == custom
            ):
                return fcn
    return None


def custom_ufunc(ufunc, layout, behavior):
    import numpy

    behavior = Behavior(ak._v2.behavior, behavior)
    custom = layout.parameter("__array__")
    if not isstr(custom):
        custom = layout.parameter("__record__")
    if not isstr(custom):
        custom = layout.purelist_parameter("__record__")
    if isstr(custom):
        for key, fcn in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and (key[0] is ufunc or key[0] is numpy.ufunc)
                and key[1] == custom
            ):
                return fcn
    return None


def numba_array_typer(layouttype, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    arr = layouttype.parameters.get("__array__")
    if isstr(arr):
        typer = behavior["__numba_typer__", arr]
        if callable(typer):
            return typer
    rec = layouttype.parameters.get("__record__")
    if isstr(rec):
        typer = behavior["__numba_typer__", ".", rec]
        if callable(typer):
            return typer
    deeprec = layouttype.parameters.get("__record__")
    if isstr(deeprec):
        typer = behavior["__numba_typer__", "*", deeprec]
        if callable(typer):
            return typer
    return None


def numba_array_lower(layouttype, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    arr = layouttype.parameters.get("__array__")
    if isstr(arr):
        lower = behavior["__numba_lower__", arr]
        if callable(lower):
            return lower
    rec = layouttype.parameters.get("__record__")
    if isstr(rec):
        lower = behavior["__numba_lower__", ".", rec]
        if callable(lower):
            return lower
    deeprec = layouttype.parameters.get("__record__")
    if isstr(deeprec):
        lower = behavior["__numba_lower__", "*", deeprec]
        if callable(lower):
            return lower
    return None


def recordclass(layout, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    rec = layout.parameter("__record__")
    if isstr(rec):
        cls = behavior[rec]
        if isinstance(cls, type) and issubclass(cls, ak._v2.highlevel.Record):
            return cls
    return ak._v2.highlevel.Record


def typestrs(behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    out = {}
    for key, typestr in behavior.items():
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "__typestr__"
            and isstr(key[1])
            and isstr(typestr)
        ):
            out[key[1]] = typestr
    return out


def gettypestr(parameters, typestrs):
    if parameters is not None:
        record = parameters.get("__record__")
        if record is not None:
            typestr = typestrs.get(record)
            if typestr is not None:
                return typestr
        array = parameters.get("__array__")
        if array is not None:
            typestr = typestrs.get(array)
            if typestr is not None:
                return typestr
    return None


def numba_record_typer(layouttype, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isstr(rec):
        typer = behavior["__numba_typer__", rec]
        if callable(typer):
            return typer
    return None


def numba_record_lower(layouttype, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isstr(rec):
        lower = behavior["__numba_lower__", rec]
        if callable(lower):
            return lower
    return None


def overload(behavior, signature):
    if not any(s is None for s in signature):
        behavior = Behavior(ak._v2.behavior, behavior)
        for key, custom in behavior.items():
            if (
                isinstance(key, tuple)
                and len(key) == len(signature)
                and key[0] == signature[0]
                and all(
                    k == s
                    or (
                        isinstance(k, type) and isinstance(s, type) and issubclass(s, k)
                    )
                    for k, s in zip(key[1:], signature[1:])
                )
            ):
                return custom


def numba_attrs(layouttype, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isstr(rec):
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
    behavior = Behavior(ak._v2.behavior, behavior)
    rec = layouttype.parameters.get("__record__")
    if isstr(rec):
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
    behavior = Behavior(ak._v2.behavior, behavior)
    done = False

    if isinstance(left, ak._v2._connect.numba.layout.ContentType):
        left = left.parameters.get("__record__")
        if not isstr(left):
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
    behavior = Behavior(ak._v2.behavior, behavior)
    done = False

    if isinstance(left, ak._v2._connect.numba.layout.ContentType):
        left = left.parameters.get("__record__")
        if not isstr(left):
            done = True

    if isinstance(right, ak._v2._connect.numba.layout.ContentType):
        right = right.parameters.get("__record__")
        if not isstr(right):
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


def behavior_of(*arrays, **kwargs):
    behavior = kwargs.get("behavior")
    if behavior is not None:
        # An explicit 'behavior' always wins.
        return behavior

    copied = False
    highs = (
        ak._v2.highlevel.Array,
        ak._v2.highlevel.Record,
        #        ak._v2.highlevel.ArrayBuilder,
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
    assert content is None or isinstance(
        content, (ak._v2.contents.Content, ak._v2.record.Record)
    )
    assert behavior is None or isinstance(behavior, Mapping)
    assert isinstance(highlevel, bool)
    if highlevel:
        if like is not None and behavior is None:
            behavior = behavior_of(like)

        if isinstance(content, ak._v2.contents.Content):
            return ak._v2.highlevel.Array(content, behavior=behavior)
        elif isinstance(content, ak._v2.record.Record):
            return ak._v2.highlevel.Record(content, behavior=behavior)

    return content


def extra(args, kwargs, defaults):
    out = []
    for i, (name, default) in enumerate(defaults):
        if i < len(args):
            out.append(args[i])
        elif name in kwargs:
            out.append(kwargs[name])
        else:
            out.append(default)
    return out


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


# def make_union(tags, index, contents, identities, parameters):
#     if isinstance(index, ak._v2.contents.Index32):
#         return ak._v2.contents.UnionArray8_32(tags, index, contents, identities, parameters)
#     elif isinstance(index, ak._v2.contents.IndexU32):
#         return ak._v2.contents.UnionArray8_U32(tags, index, contents, identities, parameters)
#     elif isinstance(index, ak._v2.index.Index64):
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
#                 ak._v2.operations.structure.fill_none(
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
#             ak._v2.index.Index64(nplike.full(len(unionarray), -1, dtype=np.int64)),
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


def direct_Content_subclass(node):
    if node is None:
        return None
    else:
        mro = type(node).mro()
        return mro[mro.index(ak._v2.contents.Content) - 1]


def direct_Content_subclass_name(node):
    out = direct_Content_subclass(node)
    if out is None:
        return None
    else:
        return out.__name__


def merge_parameters(one, two, merge_equal=False):

    if one is None and two is None:
        return None

    elif one is None:
        return two

    elif two is None:
        return one

    elif merge_equal:
        out = {}
        for k, v in two.items():
            if k in one.keys():
                if v == one[k]:
                    out[k] = v
        return out

    else:
        out = dict(one)
        for k, v in two.items():
            if v is not None:
                out[k] = v
        return out
