import pytest  # noqa: F401
import awkward._v2 as ak
from awkward._v2.operations.convert.ak_from_avro import from_avro  # noqa: F401


def test_int():
    data = [34, 45, 67, 78, 23, 89, 6, 33, 96, 73]
    assert data == ak.from_avro("int_test_data.avro")


def test_boolean():
    raise NotImplementedError


def test_long():
    raise NotImplementedError


def test_float():
    raise NotImplementedError


def test_double():
    raise NotImplementedError


def test_bytes():
    raise NotImplementedError


def test_string():
    raise NotImplementedError


def test_null():
    raise NotImplementedError
