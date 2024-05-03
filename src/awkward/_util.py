# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import base64
import os
import struct
import sys
from collections.abc import Collection

import numpy as np  # noqa: TID251
import packaging.version

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
