# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
    print()
    try:
        print("numpy.bool_: " + repr(awkward1.layout.NumpyArray(numpy.array([False, True], numpy.bool_)).format))
    except Exception as err:
        print("numpy.bool_: " + str(err))
    try:
        print("numpy.int8: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.int8)).format))
    except Exception as err:
        print("numpy.int8: " + str(err))
    try:
        print("numpy.int16: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.int16)).format))
    except Exception as err:
        print("numpy.int16: " + str(err))
    try:
        print("numpy.int32: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.int32)).format))
    except Exception as err:
        print("numpy.int32: " + str(err))
    try:
        print("numpy.int64: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.int64)).format))
    except Exception as err:
        print("numpy.int64: " + str(err))
    try:
        print("numpy.uint8: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.uint8)).format))
    except Exception as err:
        print("numpy.uint8: " + str(err))
    try:
        print("numpy.uint16: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.uint16)).format))
    except Exception as err:
        print("numpy.uint16: " + str(err))
    try:
        print("numpy.uint32: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.uint32)).format))
    except Exception as err:
        print("numpy.uint32: " + str(err))
    try:
        print("numpy.uint64: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.uint64)).format))
    except Exception as err:
        print("numpy.uint64: " + str(err))
    try:
        print("numpy.intc: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.intc)).format))
    except Exception as err:
        print("numpy.intc: " + str(err))
    try:
        print("numpy.uintc: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.uintc)).format))
    except Exception as err:
        print("numpy.uintc: " + str(err))
    try:
        print("numpy.longlong: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.longlong)).format))
    except Exception as err:
        print("numpy.longlong: " + str(err))
    try:
        print("numpy.ulonglong: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.ulonglong)).format))
    except Exception as err:
        print("numpy.ulonglong: " + str(err))
    try:
        print("numpy.float16: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.float16)).format))
    except Exception as err:
        print("numpy.float16: " + str(err))
    try:
        print("numpy.float32: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.float32)).format))
    except Exception as err:
        print("numpy.float32: " + str(err))
    try:
        print("numpy.float64: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.float64)).format))
    except Exception as err:
        print("numpy.float64: " + str(err))
    try:
        print("numpy.float128: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.float128)).format))
    except Exception as err:
        print("numpy.float128: " + str(err))
    try:
        print("numpy.complex64: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.complex64)).format))
    except Exception as err:
        print("numpy.complex64: " + str(err))
    try:
        print("numpy.complex128: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.complex128)).format))
    except Exception as err:
        print("numpy.complex128: " + str(err))
    try:
        print("numpy.complex256: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.complex256)).format))
    except Exception as err:
        print("numpy.complex256: " + str(err))
    try:
        print("numpy.datetime64: " + repr(awkward1.layout.NumpyArray(numpy.array(["2019", "2020"], numpy.datetime64)).format))
    except Exception as err:
        print("numpy.datetime64: " + str(err))
    try:
        print("numpy.timedelta64: " + repr(awkward1.layout.NumpyArray(numpy.array([1, 2, 3], numpy.timedelta64)).format))
    except Exception as err:
        print("numpy.timedelta64: " + str(err))
    try:
        print("numpy.bytes_: " + repr(awkward1.layout.NumpyArray(numpy.array([b"one", b"two"], numpy.bytes_)).format))
    except Exception as err:
        print("numpy.bytes_: " + str(err))
    try:
        print("numpy.str_: " + repr(awkward1.layout.NumpyArray(numpy.array(["one", "two"], numpy.str_)).format))
    except Exception as err:
        print("numpy.str_: " + str(err))
    try:
        print("numpy.record_: " + repr(awkward1.layout.NumpyArray(numpy.array([(1, 1.1), (2, 2.2)], [("one", int), ("two", float)])).format))
    except Exception as err:
        print("numpy.record_: " + str(err))
    try:
        print("numpy.object_: " + repr(awkward1.layout.NumpyArray(numpy.array([None, 1, "hello"], numpy.object_)).format))
    except Exception as err:
        print("numpy.object_: " + str(err))
