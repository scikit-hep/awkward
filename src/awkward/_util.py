# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import itertools
import os
import re
import sys
from collections.abc import Collection, Mapping

import packaging.version

import awkward as ak
from awkward._behavior import behavior_of
from awkward._nplikes import nplike_of
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward.typing import TypeVar

np = NumpyMetadata.instance()

win = os.name == "nt"
bits32 = np.iinfo(np.intp).bits == 32

# matches include/awkward/common.h
kMaxInt8 = 127  # 2**7  - 1
kMaxUInt8 = 255  # 2**8  - 1
kMaxInt32 = 2147483647  # 2**31 - 1
kMaxUInt32 = 4294967295  # 2**32 - 1
kMaxInt64 = 9223372036854775806  # 2**63 - 2: see below
kSliceNone = kMaxInt64 + 1  # for Slice::none()
kMaxLevels = 48


def parse_version(version):
    return packaging.version.parse(version)


def numpy_at_least(version):
    import numpy  # noqa: TID251

    return parse_version(numpy.__version__) >= parse_version(version)


def in_module(obj, modulename: str) -> bool:
    m = type(obj).__module__
    return m == modulename or m.startswith(modulename + ".")


def tobytes(array):
    if hasattr(array, "tobytes"):
        return array.tobytes()
    else:
        return array.tostring()


native_byteorder = "<" if sys.byteorder == "little" else ">"


def native_to_byteorder(array, byteorder: str):
    """
    Args:
        array: nplike array
        byteorder (`"<"` or `">"`): desired byteorder

    Return a copy of array. Swap the byteorder if `byteorder` does not match
    `ak._util.native_byteorder`. This function is _not_ idempotent; no metadata
    from `array` exists to determine its current byteorder.
    """
    assert byteorder in "<>"
    if byteorder != native_byteorder:
        return array.byteswap(inplace=False)
    else:
        return array


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


# FIXME: introduce sentinel type for this
class _Unset:
    def __repr__(self):
        return f"{__name__}.unset"


unset = _Unset()


# Sentinel object for catching pass-through values
class Unspecified:
    pass


def wrap_layout(content, behavior=None, highlevel=True, like=None, allow_other=False):
    assert (
        content is None
        or isinstance(content, (ak.contents.Content, ak.record.Record))
        or allow_other
    )
    assert behavior is None or isinstance(behavior, Mapping)
    assert isinstance(highlevel, bool)
    if highlevel:
        if like is not None and behavior is None:
            behavior = behavior_of(like)

        if isinstance(content, ak.contents.Content):
            return ak.highlevel.Array(content, behavior=behavior)
        elif isinstance(content, ak.record.Record):
            return ak.highlevel.Record(content, behavior=behavior)

    return content


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
    np = NumpyMetadata.instance()
    # overshadows global NumPy import for nplike-safety
    numpy = Numpy.instance()
    nplike = nplike_of(array)

    def recurse(array, mask=None):
        if Jax.is_tracer(array):
            raise ak._errors.wrap_error(
                TypeError("Jax tracers cannot be used with `ak.from_arraylib`")
            )

        if regulararray and len(array.shape) > 1:
            new_shape = (-1,) + array.shape[2:]
            return ak.contents.RegularArray(
                recurse(nplike.reshape(array, new_shape), mask),
                array.shape[1],
                array.shape[0],
            )

        if len(array.shape) == 0:
            array = nplike.reshape(array, (1,))

        if array.dtype.kind == "S":
            assert nplike is numpy
            asbytes = array.reshape(-1)
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = numpy.add(starts, numpy.char.str_len(asbytes))
            data = ak.contents.ListArray(
                ak.index.Index64(starts),
                ak.index.Index64(stops),
                ak.contents.NumpyArray(
                    asbytes.view("u1"),
                    parameters={"__array__": "byte"},
                    backend=ak._backends.NumpyBackend.instance(),
                ),
                parameters={"__array__": "bytestring"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak.contents.RegularArray(
                    data, array.shape[i], array.shape[i - 1]
                )

        elif array.dtype.kind == "U":
            assert nplike is numpy
            asbytes = numpy.char.encode(array.reshape(-1), "utf-8", "surrogateescape")
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = numpy.add(starts, numpy.char.str_len(asbytes))
            data = ak.contents.ListArray(
                ak.index.Index64(starts),
                ak.index.Index64(stops),
                ak.contents.NumpyArray(
                    asbytes.view("u1"),
                    parameters={"__array__": "char"},
                    backend=ak._backends.NumpyBackend.instance(),
                ),
                parameters={"__array__": "string"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak.contents.RegularArray(
                    data, array.shape[i], array.shape[i - 1]
                )

        else:
            data = ak.contents.NumpyArray(array)

        if mask is None:
            return data

        elif mask is False or (isinstance(mask, np.bool_) and not mask):
            # NumPy's MaskedArray with mask == False is an UnmaskedArray
            if len(array.shape) == 1:
                return ak.contents.UnmaskedArray(data)
            else:

                def attach(x):
                    if isinstance(x, ak.contents.NumpyArray):
                        return ak.contents.UnmaskedArray(x)
                    else:
                        return ak.contents.RegularArray(
                            attach(x.content), x.size, len(x)
                        )

                return attach(data.to_RegularArray())

        else:
            # NumPy's MaskedArray is a ByteMaskedArray with valid_when=False
            return ak.contents.ByteMaskedArray(
                ak.index.Index8(mask), data, valid_when=False
            )

    if array.dtype == np.dtype("O"):
        raise ak._errors.wrap_error(
            TypeError("Awkward Array does not support arrays with object dtypes.")
        )

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
        layout = ak.contents.RecordArray(contents, array.dtype.names)

    return ak._util.wrap_layout(layout, behavior, highlevel)


def maybe_posaxis(layout, axis, depth):
    if isinstance(layout, ak.record.Record):
        if axis == 0:
            raise ak._errors.wrap_error(
                np.AxisError("Record type at axis=0 is a scalar, not an array")
            )
        return maybe_posaxis(layout._array, axis, depth)

    if axis >= 0:
        return axis

    else:
        is_branching, additional_depth = layout.branch_depth
        if not is_branching:
            return axis + depth + additional_depth - 1
        else:
            return None


T = TypeVar("T")


def unique_list(items: Collection[T]) -> list[T]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
