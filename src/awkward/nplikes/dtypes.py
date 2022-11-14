# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

"""
The known dtypes supported by Awkward's internals, and a type-promotion table
"""
from typing import TypeVar

import numpy
from numpy import (
    bytes_,
    complexfloating,
    datetime64,
    dtype,
    floating,
    generic,
    integer,
    intp,
    longlong,
    number,
    object_,
    signedinteger,
    str_,
    timedelta64,
    unsignedinteger,
)

from awkward import _errors

__all__ = [
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "longlong",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "str_",
    "bytes_",
    "intp",
    "dtype",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "floating",
    "complexfloating",
    "number",
    "object_",
    "generic",
    "datetime64",
    "timedelta64",
]

# DTypes ###############################################################################################################
# Basic non-timelike dtypes
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
float32 = dtype("float32")
float64 = dtype("float64")
complex64 = dtype("complex64")
complex128 = dtype("complex128")
bool_ = dtype("bool")

numeric_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
)

all_dtypes = numeric_dtypes + (bool_,)

# Add missing platform-specific dtypes
if hasattr(numpy, "float16"):
    numeric_dtypes = (*numeric_dtypes, dtype(numpy.float16))
if hasattr(numpy, "complex256"):
    numeric_dtypes = (*numeric_dtypes, dtype(numpy.complex256))


def is_datetime(dtype: dtype) -> bool:
    return dtype.kind == "M"


def is_timedelta(dtype: dtype) -> bool:
    return dtype.kind == "m"


def is_timelike(dtype: dtype) -> bool:
    return is_datetime(dtype) or is_timedelta(dtype)


def is_known_dtype(dtype: dtype) -> bool:
    return dtype in all_dtypes or is_timelike(dtype)


# Promotions ###########################################################################################################
# The Array API implements special promotion rules, let's emulate them here
_non_timelike_promotion_table = {
    (int8, int8): int8,
    (int16, int8): int16,
    (int16, int16): int16,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (uint8, uint8): uint8,
    (uint16, uint8): uint16,
    (uint16, uint16): uint16,
    (uint32, uint8): uint32,
    (uint32, uint16): uint32,
    (uint32, uint32): uint32,
    (uint64, uint8): uint64,
    (uint64, uint16): uint64,
    (uint64, uint32): uint64,
    (uint64, uint64): uint64,
    (uint8, int8): int16,
    (uint8, int16): int16,
    (uint8, int32): int32,
    (uint8, int64): int64,
    (uint16, int8): int32,
    (uint16, int16): int32,
    (uint16, int32): int32,
    (uint16, int64): int64,
    (uint32, int8): int64,
    (uint32, int16): int64,
    (uint32, int32): int64,
    (uint32, int64): int64,
    (float32, float32): float32,
    (float64, float32): float64,
    (float64, float64): float64,
    (bool_, bool_): bool_,
    # Additions to Array API
    # 64 Bit
    (complex64, float32): complex64,
    (complex64, float64): complex128,
    (complex64, complex64): complex64,
    # 128 Bit
    (complex128, float32): complex128,
    (complex128, float64): complex128,
    (complex128, complex64): complex128,
    (complex128, complex128): complex128,
}


def _set_maybe_promotion():
    # Some dtypes (np.float16?, np.complex256) do not exist on all platforms
    maybe_promotion = {
        ("float128", "float32"): "float128",
        ("float128", "float64"): "float128",
        ("float128", "complex64"): "float128",
        ("float128", "float128"): "float128",
        ("complex256", "float16"): "complex256",
        ("complex256", "float32"): "complex256",
        ("complex256", "float64"): "complex256",
        ("complex256", "float128"): "complex256",
        ("complex256", "complex64"): "complex256",
        ("complex256", "complex128"): "complex256",
        ("complex256", "complex256"): "complex256",
    }

    for (left, right), result in maybe_promotion.items():
        try:
            left_type = getattr(numpy, left)
            right_type = getattr(numpy, right)
            result_type = getattr(numpy, result)
        except AttributeError:
            continue
        _non_timelike_promotion_table[dtype(left_type), dtype(right_type)] = dtype(
            result_type
        )


_set_maybe_promotion()


T = TypeVar("T")


def _permute_promotion(table: dict[tuple[T, T], T]):
    for (left, right), value in list(table.items()):
        table[right, left] = value


_permute_promotion(_non_timelike_promotion_table)


def _promote_non_timelike_types(left: dtype, right: dtype) -> dtype:
    return _non_timelike_promotion_table[left, right]


# Datetime promotions
_time_kind_ordered = ("as", "fs", "ps", "ns", "us", "ms", "s", "m", "h")
_time_kind_aliases = {"μs": "us"}
_timelike_kind_promotion_table = {
    # Date
    ("Y", "M"): "M",
    ("Y", "Y"): "Y",
    ("M", "M"): "M",
    ("W", "D"): "D",
    ("W", "W"): "W",
    ("D", "D"): "D",
    # Time
    **{
        (major, minor): minor
        for i, major in enumerate(_time_kind_ordered)
        for minor in _time_kind_ordered[: i + 1]
    },
}
_permute_promotion(_timelike_kind_promotion_table)


_timelike_promotion_table = {
    (datetime64, datetime64): datetime64,
    (timedelta64, timedelta64): timedelta64,
    (datetime64, timedelta64): datetime64,
}
_permute_promotion(_timelike_promotion_table)


def _promote_timelike_types(left: dtype, right: dtype) -> dtype:
    if not is_timelike(right):
        raise _errors.wrap_error(ValueError("expected timelike dtype"))
    try:
        kind_result = _timelike_kind_promotion_table[
            _time_kind_aliases.get(left.kind, left.kind),
            _time_kind_aliases.get(right.kind, right.kind),
        ]
    except KeyError:
        raise _errors.wrap_error(
            ValueError(f"cannot promote incompatible kinds: {left.kind} {right.kind}")
        ) from None
    type_result = _timelike_promotion_table[left.type, right.type]
    return type_result(1, kind_result).dtype


def promote_types(left: dtype, right: dtype) -> dtype:
    if is_timelike(left):
        return _promote_timelike_types(left, right)
    else:
        return _promote_non_timelike_types(left, right)
