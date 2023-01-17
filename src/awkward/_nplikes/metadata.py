# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
"""
nplike-agnostic metadata, such as dtypes and constants.
"""

from __future__ import annotations

import numbers

import numpy as _numpy
from numpy import datetime_data, dtype, finfo, iinfo

import awkward as ak

__all__ = [
    "nan",
    "inf",
    "newaxis",
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
    "intp",
    "datetime_data",
    "dtype",
    "iinfo",
    "finfo",
    "isdtype",
]


# NumPy constants
nan = _numpy.nan
inf = _numpy.inf
newaxis = _numpy.newaxis

# DTypes ######################################################################
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
intp = dtype("intp")

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
if hasattr(_numpy, "float16"):
    _real_floating = (dtype(_numpy.float16), *_real_floating)  # type: ignore
if hasattr(_numpy, "float128"):
    _real_floating = (*_real_floating, dtype(_numpy.float128))  # type: ignore
if hasattr(_numpy, "complex256"):
    _complex_floating = (*_complex_floating, dtype(_numpy.complex256))  # type: ignore


all_dtypes = (
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


# DType inspection routines ###################################################
def isdtype(
    dtype_: dtype,
    kind: dtype | str | tuple[dtype | str, ...],
) -> bool:
    if isinstance(kind, str):
        if kind == "bool":
            return _numpy.issubdtype(dtype_, bool_)
        elif kind == "signed integer":
            return any([_numpy.issubdtype(dtype_, c) for c in _signed_integer])
        elif kind == "unsigned integer":
            return any([_numpy.issubdtype(dtype_, c) for c in _unsigned_integer])
        elif kind == "integral":
            return isdtype(dtype_, "signed integer") or isdtype(
                dtype_, "unsigned integer"
            )
        elif kind == "real floating":
            return any([_numpy.issubdtype(dtype_, c) for c in _real_floating])
        elif kind == "complex floating":
            return any([_numpy.issubdtype(dtype_, c) for c in _complex_floating])
        elif kind == "numeric":
            return (
                isdtype(dtype_, "integral")
                or isdtype(dtype_, "real floating")
                or isdtype(dtype_, "complex floating")
            )
        ### Extensions to Array API ###
        elif kind == "timelike":
            return isdtype(dtype_, timedelta64) or isdtype(dtype_, datetime64)
        else:
            raise ak._errors.wrap_error(ValueError(f"Invalid kind {kind} given"))
    elif isinstance(kind, tuple):
        return any([isdtype(dtype_, k) for k in kind])
    else:
        assert isinstance(kind, dtype)
        return _numpy.issubdtype(dtype_, kind)


def default_dtype(value) -> dtype:
    if isinstance(value, numbers.Integral):
        return int64
    elif isinstance(value, numbers.Real):
        return float64
    elif isinstance(value, numbers.Complex):
        return complex128
    elif isinstance(value, (bool, _numpy.bool_)):
        return bool_
    else:
        raise ak._errors.wrap_error(TypeError("unsupported value"))


def is_valid_dtype(dtype: dtype) -> bool:
    return dtype in all_dtypes


def ensure_valid_dtype(dtype: dtype, *, allow_none=False):
    if is_valid_dtype(dtype) or (dtype is None and allow_none):
        return
    raise ak._errors.wrap_error(ValueError("dtype must be one of the supported dtypes"))
