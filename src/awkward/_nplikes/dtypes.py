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
    "dtype",
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
# Although we don't need them to be bound to names
# (Awkward code can't rely on these beign available)
# we can still allow our code to test for them
if hasattr(numpy, "float16"):
    numeric_dtypes = (*numeric_dtypes, dtype(numpy.float16))  # type: ignore
if hasattr(numpy, "complex256"):
    numeric_dtypes = (*numeric_dtypes, dtype(numpy.complex256))  # type: ignore


def is_datetime(dtype: dtype) -> bool:
    return dtype.kind == "M"


def is_timedelta(dtype: dtype) -> bool:
    return dtype.kind == "m"


def is_timelike(dtype: dtype) -> bool:
    return is_datetime(dtype) or is_timedelta(dtype)


def is_known_dtype(dtype: dtype) -> bool:
    return dtype in all_dtypes or is_timelike(dtype)


def is_string(dtype: dtype) -> bool:
    return dtype.type == numpy.string_


def is_bytes(dtype: dtype) -> bool:
    return dtype.type == numpy.bytes_
