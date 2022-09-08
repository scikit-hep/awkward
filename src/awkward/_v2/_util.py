# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

# First, transition all the _v2 code to start using implementations in this file.
# Then build up the high-level replacements.

import itertools
import numbers
import os
import re
import threading
import traceback
import packaging.version

from collections.abc import Sequence, Mapping, Iterable

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
    "jax": ak.nplike.Jax,
}


def regularize_backend(backend):
    if backend in _backends:
        return _backends[backend].instance()
    else:
        raise error(  # noqa: AK101
            ValueError("The available backends for now are `cpu` and `cuda`.")
        )


def parse_version(version):
    return packaging.version.parse(version)


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


def identifier_hash(str):
    import base64
    import struct

    return (
        base64.encodebytes(struct.pack("q", hash(str)))
        .rstrip(b"=\n")
        .replace(b"+", b"")
        .replace(b"/", b"")
        .decode("ascii")
    )


###############################################################################


class ErrorContext:
    # Any other threads should get a completely independent _slate.
    _slate = threading.local()

    _width = 80

    @classmethod
    def primary(cls):
        return cls._slate.__dict__.get("__primary_context__")

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __enter__(self):
        # Make it strictly non-reenterant. Only one ErrorContext (per thread) is primary.
        if self.primary() is None:
            self._slate.__dict__.clear()
            self._slate.__dict__.update(self._kwargs)
            self._slate.__dict__["__primary_context__"] = self

    def __exit__(self, exception_type, exception_value, traceback):
        # Step out of the way so that another ErrorContext can become primary.
        if self.primary() is self:
            self._slate.__dict__.clear()

    def format_argument(self, width, value):
        if isinstance(value, ak._v2.contents.Content):
            return self.format_argument(width, ak._v2.highlevel.Array(value))
        elif isinstance(value, ak._v2.record.Record):
            return self.format_argument(width, ak._v2.highlevel.Record(value))

        valuestr = None
        if isinstance(
            value,
            (
                ak._v2.highlevel.Array,
                ak._v2.highlevel.Record,
                ak._v2.highlevel.ArrayBuilder,
            ),
        ):
            try:
                valuestr = value._repr(width)
            except Exception as err:
                valuestr = f"repr-raised-{type(err).__name__}"

        elif value is None or isinstance(value, (bool, int, float)):
            try:
                valuestr = repr(value)
            except Exception as err:
                valuestr = f"repr-raised-{type(err).__name__}"

        elif isinstance(value, (str, bytes)):
            try:
                if len(value) < 60:
                    valuestr = repr(value)
                else:
                    valuestr = repr(value[:57]) + "..."
            except Exception as err:
                valuestr = f"repr-raised-{type(err).__name__}"

        elif isinstance(value, np.ndarray):
            import numpy

            if not numpy.__version__.startswith("1.13."):  # 'threshold' argument
                prefix = f"{type(value).__module__}.{type(value).__name__}("
                suffix = ")"
                try:
                    valuestr = numpy.array2string(
                        value,
                        max_line_width=width - len(prefix) - len(suffix),
                        threshold=0,
                    ).replace("\n", " ")
                    valuestr = prefix + valuestr + suffix
                except Exception as err:
                    valuestr = f"array2string-raised-{type(err).__name__}"

                if len(valuestr) > width and "..." in valuestr[:-1]:
                    last = valuestr.rfind("...") + 3
                    while last > width:
                        last = valuestr[: last - 3].rfind("...") + 3
                    valuestr = valuestr[:last]

                if len(valuestr) > width:
                    valuestr = valuestr[: width - 3] + "..."

        elif isinstance(value, (Sequence, Mapping)) and len(value) < 10000:
            valuestr = repr(value)
            if len(valuestr) > width:
                valuestr = valuestr[: width - 3] + "..."

        if valuestr is None:
            return f"{type(value).__name__}-instance"
        else:
            return valuestr


class OperationErrorContext(ErrorContext):
    def __init__(self, name, arguments):
        string_arguments = {}
        for key, value in arguments.items():
            if isstr(key):
                width = self._width - 8 - len(key) - 3
            else:
                width = self._width - 8

            string_arguments[key] = self.format_argument(width, value)

        super().__init__(
            name=name,
            arguments=string_arguments,
            traceback=traceback.extract_stack(limit=3)[0],
        )

    @property
    def name(self):
        return self._kwargs["name"]

    @property
    def arguments(self):
        return self._kwargs["arguments"]

    @property
    def traceback(self):
        return self._kwargs["traceback"]

    def format_exception(self, exception):
        tb = self.traceback
        try:
            location = f" (from {tb.filename}, line {tb.lineno})"
        except Exception:
            location = ""

        arguments = []
        for name, valuestr in self.arguments.items():
            if isstr(name):
                arguments.append(f"\n        {name} = {valuestr}")
            else:
                arguments.append(f"\n        {valuestr}")

        extra_line = "" if len(arguments) == 0 else "\n    "
        return f"""while calling{location}

    {self.name}({"".join(arguments)}{extra_line})

Error details: {str(exception)}"""


class SlicingErrorContext(ErrorContext):
    def __init__(self, array, where):
        super().__init__(
            array=self.format_argument(self._width - 4, array),
            where=self.format_slice(where),
            traceback=traceback.extract_stack(limit=3)[0],
        )

    @property
    def array(self):
        return self._kwargs["array"]

    @property
    def where(self):
        return self._kwargs["where"]

    @property
    def traceback(self):
        return self._kwargs["traceback"]

    def format_exception(self, exception):
        tb = self.traceback
        try:
            location = f" (from {tb.filename}, line {tb.lineno})"
        except Exception:
            location = ""

        if isstr(exception):
            message = exception
        else:
            message = f"Error details: {str(exception)}"

        return f"""while attempting to slice{location}

    {self.array}

with

    {self.where}

{message}"""

    @staticmethod
    def format_slice(x):
        if isinstance(x, slice):
            if x.step is None:
                return "{}:{}".format(
                    "" if x.start is None else x.start,
                    "" if x.stop is None else x.stop,
                )
            else:
                return "{}:{}:{}".format(
                    "" if x.start is None else x.start,
                    "" if x.stop is None else x.stop,
                    x.step,
                )

        elif isinstance(x, tuple):
            return "(" + ", ".join(SlicingErrorContext.format_slice(y) for y in x) + ")"

        elif isinstance(x, ak._v2.index.Index64):
            return str(x.data)

        elif isinstance(x, ak._v2.contents.Content):
            try:
                return str(ak._v2.highlevel.Array(x))
            except Exception:
                return x._repr("    ", "", "")

        elif isinstance(x, ak._v2.record.Record):
            try:
                return str(ak._v2.highlevel.Record(x))
            except Exception:
                return x._repr("    ", "", "")

        else:
            return repr(x)


def error(exception, error_context=None):
    if isinstance(exception, type) and issubclass(exception, Exception):
        try:
            exception = exception()
        except Exception:
            return exception

    if isinstance(exception, (NotImplementedError, AssertionError)):
        return type(exception)(
            str(exception)
            + "\n\nSee if this has been reported at https://github.com/scikit-hep/awkward-1.0/issues"
        )

    if error_context is None:
        error_context = ErrorContext.primary()

    if isinstance(error_context, ErrorContext):
        # Note: returns an error for the caller to raise!
        return type(exception)(error_context.format_exception(exception))
    else:
        # Note: returns an error for the caller to raise!
        return exception


def indexerror(subarray, slicer, details=None):
    detailsstr = ""
    if details is not None:
        detailsstr = f"""

Error details: {details}."""

    error_context = ErrorContext.primary()
    if not isinstance(error_context, SlicingErrorContext):
        # Note: returns an error for the caller to raise!
        return IndexError(
            f"cannot slice {type(subarray).__name__} with {SlicingErrorContext.format_slice(slicer)}{detailsstr}"
        )

    else:
        # Note: returns an error for the caller to raise!
        return IndexError(
            error_context.format_exception(
                f"at inner {type(subarray).__name__} of length {subarray.length}, using sub-slice {error_context.format_slice(slicer)}.{detailsstr}"
            )
        )


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


def reducer_recordclass(reducer, layout, behavior):
    behavior = Behavior(ak._v2.behavior, behavior)
    rec = layout.parameter("__record__")
    if isstr(rec):
        return behavior[reducer.highlevel_function(), rec]


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
#         raise error(ValueError(
#             "key {0} not found in record".format(repr(key))
#         ))
#     else:
#         return attempt


# key2index._pattern = re.compile(r"^[1-9][0-9]*$")


# def make_union(tags, index, contents, identifier, parameters):
#     if isinstance(index, ak._v2.contents.Index32):
#         return ak._v2.contents.UnionArray8_32(tags, index, contents, identities, parameters)
#     elif isinstance(index, ak._v2.contents.IndexU32):
#         return ak._v2.contents.UnionArray8_U32(tags, index, contents, identities, parameters)
#     elif isinstance(index, ak._v2.index.Index64):
#         return ak._v2.contents.UnionArray8_64(tags, index, contents, identities, parameters)
#     else:
#         raise error(AssertionError(index))


def union_to_record(unionarray, anonymous):
    nplike = ak.nplike.of(unionarray)

    contents = []
    for layout in unionarray.contents:
        if layout.is_IndexedType and not layout.is_OptionType:
            contents.append(layout.project())
        elif layout.is_UnionType:
            contents.append(union_to_record(layout, anonymous))
        elif layout.is_OptionType:
            contents.append(
                ak._v2.operations.fill_none(layout, np.nan, axis=0, highlevel=False)
            )
        else:
            contents.append(layout)

    if not any(isinstance(x, ak._v2.contents.RecordArray) for x in contents):
        return ak._v2.contents.UnionArray(
            unionarray.tags,
            unionarray.index,
            contents,
            unionarray.identifier,
            unionarray.parameters,
        )

    else:
        seen = set()
        all_names = []
        for layout in contents:
            if isinstance(layout, ak._v2.contents.RecordArray):
                for field in layout.fields:
                    if field not in seen:
                        seen.add(field)
                        all_names.append(field)
            else:
                if anonymous not in seen:
                    seen.add(anonymous)
                    all_names.append(anonymous)

        missingarray = ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(nplike.full(len(unionarray), -1, dtype=np.int64)),
            ak._v2.contents.EmptyArray(),
        )

        all_fields = []
        for name in all_names:
            union_contents = []
            for layout in contents:
                if isinstance(layout, ak._v2.contents.RecordArray):
                    for field in layout.fields:
                        if name == field:
                            union_contents.append(layout._getitem_field(field))
                            break
                    else:
                        union_contents.append(missingarray)
                else:
                    if name == anonymous:
                        union_contents.append(layout)
                    else:
                        union_contents.append(missingarray)

            all_fields.append(
                ak._v2.contents.UnionArray(
                    unionarray.tags,
                    unionarray.index,
                    union_contents,
                    unionarray.identifier,
                    unionarray.parameters,
                ).simplify_uniontype()
            )

        return ak._v2.contents.RecordArray(all_fields, all_names, len(unionarray))


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


def expand_braces(text, seen=None):
    if seen is None:
        seen = set()

    spans = [m.span() for m in expand_braces.regex.finditer(text)][::-1]
    alts = [text[start + 1 : stop - 1].split(",") for start, stop in spans]

    if len(spans) == 0:
        if text not in seen:
            yield text
        seen.add(text)

    else:
        for combo in itertools.product(*alts):
            replaced = list(text)
            for (start, stop), replacement in zip(spans, combo):
                replaced[start:stop] = replacement
            yield from expand_braces("".join(replaced), seen)


expand_braces.regex = re.compile(r"\{[^\{\}]*\}")


def from_arraylib(array, regulararray, recordarray, highlevel, behavior):
    np = ak.nplike.NumpyMetadata.instance()
    numpy = ak.nplike.Numpy.instance()

    def recurse(array, mask=None):
        if regulararray and len(array.shape) > 1:
            return ak._v2.contents.RegularArray(
                recurse(array.reshape((-1,) + array.shape[2:])),
                array.shape[1],
                array.shape[0],
            )

        if len(array.shape) == 0:
            array = ak._v2.contents.NumpyArray(array.reshape(1))

        if array.dtype.kind == "S":
            asbytes = array.reshape(-1)
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = starts + numpy.char.str_len(asbytes)
            data = ak._v2.contents.ListArray(
                ak._v2.index.Index64(starts),
                ak._v2.index.Index64(stops),
                ak._v2.contents.NumpyArray(
                    asbytes.view("u1"), parameters={"__array__": "byte"}, nplike=numpy
                ),
                parameters={"__array__": "bytestring"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak._v2.contents.RegularArray(
                    data, array.shape[i], array.shape[i - 1]
                )

        elif array.dtype.kind == "U":
            asbytes = numpy.char.encode(array.reshape(-1), "utf-8", "surrogateescape")
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = starts + numpy.char.str_len(asbytes)
            data = ak._v2.contents.ListArray(
                ak._v2.index.Index64(starts),
                ak._v2.index.Index64(stops),
                ak._v2.contents.NumpyArray(
                    asbytes.view("u1"), parameters={"__array__": "char"}, nplike=numpy
                ),
                parameters={"__array__": "string"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak._v2.contents.RegularArray(
                    data, array.shape[i], array.shape[i - 1]
                )

        else:
            data = ak._v2.contents.NumpyArray(array)

        if mask is None:
            return data

        elif mask is False or (isinstance(mask, np.bool_) and not mask):
            # NumPy's MaskedArray with mask == False is an UnmaskedArray
            if len(array.shape) == 1:
                return ak._v2.contents.UnmaskedArray(data)
            else:

                def attach(x):
                    if isinstance(x, ak._v2.contents.NumpyArray):
                        return ak._v2.contents.UnmaskedArray(x)
                    else:
                        return ak._v2.contents.RegularArray(
                            attach(x.content), x.size, len(x)
                        )

                return attach(data.toRegularArray())

        else:
            # NumPy's MaskedArray is a ByteMaskedArray with valid_when=False
            return ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(mask), data, valid_when=False
            )

        return data

    if isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmask(array)
        array = numpy.ma.getdata(array)
        if isinstance(mask, np.ndarray) and len(mask.shape) > 1:
            regulararray = True
            mask = mask.reshape(-1)
    else:
        mask = None

    if not recordarray or array.dtype.names is None:
        layout = recurse(array, mask)

    else:
        contents = []
        for name in array.dtype.names:
            contents.append(recurse(array[name], mask))
        layout = ak._v2.contents.RecordArray(contents, array.dtype.names)

    return ak._v2._util.wrap(layout, behavior, highlevel)


def to_arraylib(module, array, allow_missing):
    def _impl(array):
        if isinstance(array, (bool, numbers.Number)):
            return module.array(array)

        elif isinstance(array, module.ndarray):
            return array

        elif isinstance(array, np.ndarray):
            return module.asarray(array)

        elif isinstance(array, ak._v2.highlevel.Array):
            return _impl(array.layout)

        elif isinstance(array, ak._v2.highlevel.Record):
            raise ak._v2._util.error(
                ValueError(f"{module.__name__} does not support record structures")
            )

        elif isinstance(array, ak._v2.highlevel.ArrayBuilder):
            return _impl(array.snapshot().layout)

        elif isinstance(array, ak.layout.ArrayBuilder):
            return _impl(array.snapshot())

        elif ak._v2.operations.parameters(array).get("__array__") in (
            "bytestring",
            "string",
        ):
            raise ak._v2._util.error(
                ValueError(f"{module.__name__} does not support arrays of strings")
            )

        elif isinstance(array, ak._v2.contents.EmptyArray):
            return module.array([])

        elif isinstance(array, ak._v2.contents.IndexedArray):
            return _impl(array.project())

        elif isinstance(array, ak._v2.contents.UnionArray):
            contents = [_impl(array.project(i)) for i in range(len(array.contents))]
            out = module.concatenate(contents)

            tags = module.asarray(array.tags)
            for tag, content in enumerate(contents):
                mask = tags == tag
                if type(out).__module__.startswith("jaxlib."):
                    out = out.at[mask].set(content)
                else:
                    out[mask] = content
            return out

        elif isinstance(array, ak._v2.contents.UnmaskedArray):
            return _impl(array.content)

        elif isinstance(array, ak._v2.contents.IndexedOptionArray):
            content = _impl(array.project())

            mask0 = array.mask_as_bool(valid_when=False)
            if mask0.any():
                raise ak._v2._util.error(
                    ValueError(f"{module.__name__} does not support masked arrays")
                )
            else:
                return content

        elif isinstance(array, ak._v2.contents.RegularArray):
            out = _impl(array.content)
            head, tail = out.shape[0], out.shape[1:]
            shape = (head // array.size, array.size) + tail
            return out[: shape[0] * array.size].reshape(shape)

        elif isinstance(
            array, (ak._v2.contents.ListArray, ak._v2.contents.ListOffsetArray)
        ):
            return _impl(array.toRegularArray())

        elif isinstance(array, ak._v2.contents.recordarray.RecordArray):
            raise ak._v2._util.error(
                ValueError(f"{module.__name__} does not support record structures")
            )

        elif isinstance(array, ak._v2.contents.NumpyArray):
            return module.asarray(array.data)

        elif isinstance(array, ak._v2.contents.Content):
            raise ak._v2._util.error(
                AssertionError(f"unrecognized Content type: {type(array)}")
            )

        elif isinstance(array, Iterable):
            return module.asarray(array)

        else:
            raise ak._v2._util.error(
                ValueError(f"cannot convert {array} into {type(module.array([]))}")
            )

    if module.__name__ in ("jax.numpy", "cupy"):
        return _impl(array)
    elif module.__name__ == "numpy":
        layout = ak._v2.operations.to_layout(array, allow_record=True, allow_other=True)

        if isinstance(layout, (ak._v2.contents.Content, ak._v2.record.Record)):
            return layout.to_numpy(allow_missing=allow_missing)
        else:
            return module.asarray(array)
    else:
        raise ak._v2._util.error(
            ValueError(f"{module.__name__} is not supported by to_arraylib")
        )
