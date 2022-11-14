# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

"""
The known dtypes supported by Awkward's internals, and a type-promotion table
"""
import numpy
from numpy import (  # Categories
    bool_,
    bytes_,
    complex64,
    complex128,
    complexfloating,
    datetime64,
    dtype,
    float32,
    float64,
    floating,
    generic,
    int8,
    int16,
    int32,
    int64,
    integer,
    intp,
    longlong,
    number,
    object_,
    signedinteger,
    str_,
    timedelta64,
    uint8,
    uint16,
    uint32,
    uint64,
    unsignedinteger,
)

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


# The Array API implements special promotion rules, let's emulate them here
_promotion_table = {
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
    (bool, bool): bool,
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
        _promotion_table[left_type, right_type] = result_type


_set_maybe_promotion()


def _permute_promotion():
    for (left, right), value in _promotion_table.items():
        _promotion_table[right, left] = value


_permute_promotion()


def promote_types(left, right):
    return _promotion_table[left, right]
