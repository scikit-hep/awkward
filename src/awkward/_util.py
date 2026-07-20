# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import base64
import numbers
import os
import re
import struct
import sys
from collections.abc import Collection

import numpy as np  # noqa: TID251
import packaging.version

import awkward as ak
from awkward._typing import TypeVar

win = os.name == "nt"
bits32 = struct.calcsize("P") * 8 == 32
numpy2 = packaging.version.parse(np.__version__) >= packaging.version.Version("2.0.0b1")


# matches include/awkward/common.h
kMaxInt8 = 127  # 2**7  - 1
kMaxUInt8 = 255  # 2**8  - 1
kMaxInt32 = 2147483647  # 2**31 - 1
kMaxUInt32 = 4294967295  # 2**32 - 1
kMaxInt64 = 9223372036854775806  # 2**63 - 2: see below
kSliceNone = kMaxInt64 + 1  # for Slice::none()
kMaxLevels = 48


_memory_size_pattern = re.compile(
    r"^\s*([+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?)\s*([kmgtpezy]?i?b)\s*$",
    re.I,
)

_memory_size_units = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "PB": 1000**5,
    "EB": 1000**6,
    "ZB": 1000**7,
    "YB": 1000**8,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
    "PIB": 1024**5,
    "EIB": 1024**6,
    "ZIB": 1024**7,
    "YIB": 1024**8,
}


def parse_memory_size(data) -> int:
    """
    Regularizes strings like ``'## kB'`` and plain integer numbers of bytes to
    an integer number of bytes.

    Both decimal (``kB``, ``MB``, ``GB``, ...) and binary (``kiB``, ``MiB``,
    ``GiB``, ...) unit prefixes are recognized, case-insensitively.
    """
    if isinstance(data, str):
        match = _memory_size_pattern.match(data)
        if match is not None:
            target, unit = float(match.group(1)), match.group(5).upper()
            return int(target * _memory_size_units[unit])

    elif not isinstance(data, bool) and isinstance(data, numbers.Integral):
        return int(data)

    raise TypeError(
        "a number of bytes or a memory-size string with units "
        f"(such as '100 MiB') is required, not {data!r}"
    )


def in_module(obj, modulename: str) -> bool:
    m = type(obj).__module__
    return m == modulename or m.startswith(modulename + ".")


def tobytes(array):
    return array.tobytes()


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
        return ak._nplikes.dispatch.nplike_of_obj(array).byteswap(array)
    else:
        return array


def identifier_hash(str):
    return (
        base64.encodebytes(struct.pack("q", hash(str)))
        .rstrip(b"=\n")
        .replace(b"+", b"")
        .replace(b"/", b"")
        .decode("ascii")
    )


class Sentinel:
    """A class for implementing sentinel types"""

    def __init__(self, name, module=None):
        self._name = name
        self._module = module

    def __repr__(self):
        if self._module is not None:
            return f"{self._module}.{self._name}"
        else:
            return f"{self._name}"


UNSET = Sentinel("UNSET", __name__)

STDOUT = Sentinel("STDOUT", __name__)
STDOUT.stream = sys.stdout


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


def copy_behaviors(from_name: str, to_name: str, behavior: dict):
    output = {}

    for key, value in behavior.items():
        if isinstance(key, str):
            if key == from_name:
                output[to_name] = value
        else:
            if from_name in key:
                new_tuple = tuple(to_name if k == from_name else k for k in key)
                output[new_tuple] = value

    return output


def maybe_shape_of(
    obj: ak._nplikes.ArrayLike,
) -> tuple[ak._nplikes.shape.ShapeItem, ...]:
    """
    Gets the shape of an object while keeping `unknown_length` in the shape tuple.
    Virtual arrays `.shape` property materializes the shape so we use this function
    to get the shape of objects without materializing it in the case of virtual arrays.
    Unknown dimensions will be represted as `unknown_length`.
    """
    if isinstance(obj, ak._nplikes.virtual.VirtualNDArray):
        return obj._shape
    else:
        return obj.shape


def maybe_length_of(
    obj: ak.contents.Content | ak.index.Index,
) -> ak._nplikes.shape.ShapeItem:
    """
    Gets the length of an object if it is known, otherwise returns `unknown_length`.
    We use this function to get the length of objects without materializing the underlying
    virtual array buffers' shape.
    Useful for example in `__init__` methods and `__repr__` where we don't want to
    materialize the virtual array shapes.
    """
    if isinstance(obj, (ak.contents.NumpyArray, ak.index.Index)):
        return maybe_shape_of(obj._data)[0]
    elif isinstance(obj, ak.contents.ListOffsetArray):
        return maybe_length_of(obj._offsets) - 1
    elif isinstance(obj, ak.contents.ListArray):
        return maybe_length_of(obj._starts)
    elif isinstance(
        obj,
        (ak.contents.RecordArray, ak.contents.RegularArray, ak.contents.BitMaskedArray),
    ):
        return obj._length
    elif isinstance(obj, ak.contents.ByteMaskedArray):
        return maybe_length_of(obj._mask)
    elif isinstance(obj, (ak.contents.IndexedArray, ak.contents.IndexedOptionArray)):
        return maybe_length_of(obj._index)
    elif isinstance(obj, ak.contents.UnionArray):
        return maybe_length_of(obj._tags)
    elif isinstance(obj, ak.contents.UnmaskedArray):
        return maybe_length_of(obj._content)
    elif isinstance(obj, ak.contents.EmptyArray):
        return 0
    else:
        raise TypeError(f"Invalid type {type(obj)} for length calculation")
