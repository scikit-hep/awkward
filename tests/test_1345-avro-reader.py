import pytest  # noqa: F401
import awkward._v2 as ak
from awkward._v2.operations.convert.ak_from_avro import from_avro_file  # noqa: F401


def test_int():
    data = [34, 45, 67, 78, 23, 89, 6, 33, 96, 73]
    assert (
        ak.from_avro_file(
            file_name="tests/samples/int_test_data.avro", reader_lang="py"
        ).to_list()
        == data
    )


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
