# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

"""
The known dtypes supported by Awkward's internals, and a type-promotion table
"""

import numpy
from numpy import dtype

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
    "float32",
    "float64",
    "complex64",
    "complex128",
    "datetime64",
    "timedelta64",
    "dtype",
]

# DTypes #######################
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
str_ = dtype("str_")
bytes_ = dtype("bytes_")
datetime64 = dtype("datetime64")
timedelta64 = dtype("timedelta64")

_signed_integer = (int8, int16, int32, int64)
_unsigned_integer = (
    uint8,
    uint16,
    uint32,
    uint64,
)
_real_floating = (
    float32,
    float64,
)
_complex_floating = (
    complex64,
    complex128,
)


# Add missing platform-specific dtypes
# Although we don't need them to be bound to names
# (Awkward code can't rely on these being available)
# we can still allow our code to test for them
if hasattr(numpy, "float16"):
    _real_floating = (dtype(numpy.float16), *_real_floating)  # type: ignore
if hasattr(numpy, "float128"):
    _real_floating = (*_real_floating, dtype(numpy.float128))  # type: ignore
if hasattr(numpy, "complex256"):
    _complex_floating = (*_complex_floating, dtype(numpy.complex256))  # type: ignore


_all_dtypes = (
    *_signed_integer,
    *_unsigned_integer,
    *_real_floating,
    *_complex_floating,
    str_,
    bytes_,
    bool_,
    timedelta64,
    datetime64,
)
